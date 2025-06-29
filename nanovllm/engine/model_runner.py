import pickle
import time
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config, PPNodeType
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.deepseek_v3 import DeepseekV3ForCausalLLMFirst, DeepseekV3ForCausalLLMMiddle, DeepseekV3ForCausalLLMLast
                                        
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context, Context, set_tp_context_info
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, tp_rank: int, event: Event | list[Event]):
        
        config.local_rank = tp_rank

        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.tp_world_size = config.tensor_parallel_size
        self.world_size = config.node_num * config.tensor_parallel_size
        self.rank = config.node_id * config.tensor_parallel_size + tp_rank
        self.tp_rank = tp_rank
        self.event = event
        self.device = torch.device("cuda", self.tp_rank)
    
        print("waiting dist init")
        dist.init_process_group("nccl", "tcp://10.18.17.159:2333", world_size=self.world_size, rank=self.rank)
        print("dist init done")

        self.tp_groups = []
        for tp_group_idx in range(config.node_num):
            self.tp_groups.append(dist.new_group(ranks=[tp_group_idx * config.tensor_parallel_size + i for i in range(config.tensor_parallel_size)]))        
        self.tp_group = self.tp_groups[config.node_id]
        set_tp_context_info(self.tp_group, tp_rank, config.tensor_parallel_size)

        # self.pp_groups = []
        # # make a circle, chain the first rank in each node
        # for pp_group_idx in range(config.node_num):
        #     rank_a = pp_group_idx * config.tensor_parallel_size
        #     rank_b = ((pp_group_idx + 1) * config.tensor_parallel_size) % self.world_size
        #     self.pp_groups.append(dist.new_group(ranks=[rank_a, rank_b]))
        
        # self.prev_pp_group = self.pp_groups[(config.node_id - 1) % config.node_num]
        # self.next_pp_group = self.pp_groups[config.node_id]
        self.prev_pp_head_node_global_rank = ((config.node_id - 1) * config.tensor_parallel_size) % self.world_size
        self.next_pp_head_node_global_rank = ((config.node_id + 1) * config.tensor_parallel_size) % self.world_size
        

        pp_start_layer_id, pp_end_layer_id, pp_node_type = config.pp_schema


        torch.cuda.set_device(tp_rank)
        default_dtype = torch.get_default_dtype()
        
        assert hf_config.torch_dtype == torch.float16 # TODO: FIXME: hardcoded for z100

        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device(self.device)
        
        if "deepseek" in hf_config.__class__.__name__.lower():
            if pp_node_type == PPNodeType.PPNodeFirst:
                self.model = DeepseekV3ForCausalLLMFirst(hf_config, config)
            elif pp_node_type == PPNodeType.PPNodeMiddle:
                self.model = DeepseekV3ForCausalLLMMiddle(hf_config, config)
            elif pp_node_type == PPNodeType.PPNodeLast:
                self.model = DeepseekV3ForCausalLLMLast(hf_config, config)
            else:
                raise Exception
        else:
            self.model = Qwen3ForCausalLM(hf_config)

        # import pdb; pdb.set_trace()

        
        # load_model(self.model, config.model, tp_size=config.tensor_parallel_size, local_rank=tp_rank, start_layer=pp_start_layer_id, end_layer=pp_end_layer_id)
        self.pp_node_type = pp_node_type


        print(f"{time.time()}, rank{self.tp_rank}, in ModelRunner __init__ before global barrier")
        dist.barrier()
        print(f"{time.time()}, rank{self.tp_rank}, in ModelRunner __init__ after global barrier")


        self.sampler = Sampler()
        self.allocate_kv_cache(config.gpu_memory_utilization)
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        
        if self.tp_world_size > 1:
            if tp_rank == 0:
                try:
                    self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                except FileExistsError:
                    self.shm = SharedMemory(name="nanovllm", size=2**20)

                dist.barrier(group=self.tp_group)
                
                # only the first node's first process doesn't run the loop.
                if(pp_node_type != PPNodeType.PPNodeFirst):
                    self.loop()
            else:
                dist.barrier(group=self.tp_group)
                self.shm = SharedMemory(name="nanovllm")
                self.loop()


    def exit(self):
        if self.tp_world_size > 1:
            self.shm.close()
            dist.barrier(group=self.tp_group)
            if self.tp_rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()


    def loop(self):
        if True:
            prof_early_break_counter = 0
            with torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=10, active=300),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
            ) as prof:
                while True:
                    continue_loop = self.loop_inner()
                    if not continue_loop:
                        break

                    prof_early_break_counter += 1
                    if prof_early_break_counter > 150:
                        break
                    prof.step()
                    
            prof.export_chrome_trace(f"tracing-node-{self.config.node_id}-rank-{self.tp_rank}.json.gz")
        else:
            while True:
                continue_loop = self.loop_inner()
                if not continue_loop:
                    break


    def loop_inner(self):
        if self.tp_rank == 0:
            # for tp_rank 0 in nonn-first layer, it should receive request from previous node and dispatch them to local tp ranks.
            assert self.pp_node_type != PPNodeType.PPNodeFirst
            
            
            context, hidden_state, positions = self.recv_pp_cmd()

            # print(f"{time.time()}, rank{self.tp_rank}, in loop(), get new pp cmd, context={context}, hidden_state={hidden_state}, positions={positions}")

            args = (hidden_state, positions, context)
            self.call("run_non_first_node", *args)
            return True

        else:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                return False
            return True

    def send_pp_cmd(self, context, hidden_state, positions):
        assert self.tp_rank == 0
        if self.pp_node_type != PPNodeType.PPNodeLast:

            # print(f"{time.time()}, rank{self.tp_rank}, in send_pp_cmd(), slot_mapping={context.slot_mapping.size()}, context_lens={context.context_lens.size()}, block_tables={context.block_tables.size()}, hidden_state={hidden_state.size()}, positions={positions.size()}")
            with torch.profiler.record_function("create meta tensor"):
                meta_tensor = torch.tensor([
                        int(context.is_prefill),
                        context.slot_mapping.size()[0],
                        context.context_lens.size()[0],
                        context.block_tables.size()[0],
                        context.block_tables.size()[1],
                        hidden_state.size()[0],
                        positions.size()[0]
                    ], dtype=torch.int32, device=self.device)
            
            # import pdb;pdb.set_trace()
            # print(f"{time.time()}, rank{self.tp_rank}, in send_pp_cmd(), meta_tensor={meta_tensor}")

            # print(f"{time.time()}, rank{self.tp_rank}, dist.send size={meta_tensor.size()}, dtype={meta_tensor.dtype} meta_tensor")
            dist.send(meta_tensor, self.next_pp_head_node_global_rank)

            # print(f"{time.time()}, rank{self.tp_rank}, dist.send size={context.slot_mapping.size()}, dtype={context.slot_mapping.dtype} slot_mapping")
            dist.send(context.slot_mapping, self.next_pp_head_node_global_rank)

            # print(f"{time.time()}, rank{self.tp_rank}, dist.send size={context.context_lens.size()}, dtype={context.context_lens.dtype} context_lens")
            dist.send(context.context_lens, self.next_pp_head_node_global_rank)

            # print(f"{time.time()}, rank{self.tp_rank}, dist.send size={context.block_tables.size()}, dtype={context.block_tables.dtype} block_tables")
            dist.send(context.block_tables, self.next_pp_head_node_global_rank)

            # print(f"{time.time()}, rank{self.tp_rank}, dist.send size={hidden_state.size()}, dtype={hidden_state.dtype} hidden_state")
            dist.send(hidden_state, self.next_pp_head_node_global_rank)

            # print(f"{time.time()}, rank{self.tp_rank}, dist.send size={positions.size()}, dtype={positions.dtype} positions")
            dist.send(positions, self.next_pp_head_node_global_rank)
        else:
            # print(f"{time.time()}, rank{self.tp_rank}, dist.send size={hidden_state.size()}, dtype={hidden_state.dtype} hidden_state")
            dist.send(hidden_state, self.next_pp_head_node_global_rank)

    def recv_pp_cmd(self, logit_tensor=None):
        assert self.tp_rank == 0
        if self.pp_node_type != PPNodeType.PPNodeFirst:
            meta_tensor = torch.zeros([7, ], dtype=torch.int32, device=self.device)

            # print(f"{time.time()}, rank{self.tp_rank}, dist.recv size={meta_tensor.size()}, dtype={meta_tensor.dtype} meta_tensor")
            dist.recv(meta_tensor, self.prev_pp_head_node_global_rank)


            # print(f"{time.time()}, rank{self.tp_rank}, in recv_pp_cmd(), meta_tensor={meta_tensor}")

            is_prefill = bool(int(meta_tensor[0]))
            slot_mapping = torch.empty([int(meta_tensor[1]),], dtype=torch.int32, device=self.device)
            context_lens = torch.empty([int(meta_tensor[2]),], dtype=torch.int32, device=self.device)
            block_tables = torch.empty([int(meta_tensor[3]), int(meta_tensor[4])], dtype=torch.int32, device=self.device)
            hidden_state = torch.empty([int(meta_tensor[5]), self.config.hf_config.hidden_size], dtype=torch.float16, device=self.device)
            positions = torch.empty([int(meta_tensor[6]),], dtype=torch.int64, device=self.device)

            # print(f"{time.time()}, rank{self.tp_rank}, dist.recv size={slot_mapping.size()}, dtype={slot_mapping.dtype} slot_mapping")
            dist.recv(slot_mapping, self.prev_pp_head_node_global_rank)

            # print(f"{time.time()}, rank{self.tp_rank}, dist.recv size={context_lens.size()}, dtype={context_lens.dtype} context_lens")
            dist.recv(context_lens, self.prev_pp_head_node_global_rank)

            # print(f"{time.time()}, rank{self.tp_rank}, dist.recv size={block_tables.size()}, dtype={block_tables.dtype} block_tables")
            dist.recv(block_tables, self.prev_pp_head_node_global_rank)

            # print(f"{time.time()}, rank{self.tp_rank}, dist.recv size={hidden_state.size()}, dtype={hidden_state.dtype} hidden_state")
            dist.recv(hidden_state, self.prev_pp_head_node_global_rank)

            # print(f"{time.time()}, rank{self.tp_rank}, dist.recv size={positions.size()}, dtype={positions.dtype} positions")
            dist.recv(positions, self.prev_pp_head_node_global_rank)

            # # TODO: FIXME: it seems a bug here, the dist.recv() can't work with tensor on CPU, so we must put it on GPU.
            # # but when index something like freqs_cis, it should on CPU (or, it should on different GPU, but when dispatch it by shared memory, all rank will get gpu0)
            # positions = positions.cpu()

            # TODO: FIXME: hard code to None and 0 since not support batch and cumulate now
            context = Context(is_prefill, None, None, 0, 0, slot_mapping, context_lens, block_tables)
            return context, hidden_state, positions
        else:
            assert logit_tensor is not None
            # print(f"{time.time()}, rank{self.tp_rank}, dist.recv size={logit_tensor.size()}, dtype={logit_tensor.dtype} logit_tensor")
            dist.recv(logit_tensor, self.prev_pp_head_node_global_rank)
            
            


    def read_shm(self):
        assert self.tp_world_size > 1 and self.tp_rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.tp_world_size > 1 and not self.tp_rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        assert n + 4 <= self.shm.size
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.tp_world_size > 1 and self.tp_rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        assert callable(method)
        return method(*args)

    def allocate_kv_cache(self, gpu_memory_utilization):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        # TODO: harded coded for deepseek v3 and MLA
        num_kv_heads = 1
        # num_kv_heads = hf_config.num_key_value_heads // self.tp_world_size

        kv_cache_head_dim = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim 
        block_bytes = 1 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * kv_cache_head_dim * hf_config.torch_dtype.itemsize
        # block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        
        config.num_kvcache_blocks = int(total * gpu_memory_utilization - used) // block_bytes
        
        # TODO：harded coded for deepseek v3 and MLA
        self.kv_cache = torch.zeros(hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, kv_cache_head_dim)
        # self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        
        layer_id = 0
        for module in self.model.modules():
            # if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            #     module.k_cache = self.kv_cache[0, layer_id]
            #     module.v_cache = self.kv_cache[1, layer_id]
            #     layer_id += 1
            if hasattr(module, "kv_c_and_k_pe_cache"):
                module.k_cache = self.kv_cache[layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        context_lens = [] # for prefill that does not support cumulative input 
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            context_lens.append(seqlen)
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        assert len(input_ids) == len(slot_mapping)

        # TODO: FIXME: hardcoded for PP to make cross node comm easy, make block_tables never none. disable prefix cache
        block_tables = self.prepare_block_tables(seqs)
        # if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
        #     block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(device=self.device, non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(device=self.device, non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(device=self.device, non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(device=self.device, non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(device=self.device, non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(device=self.device, non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        
        # print(f"{time.time()}, rank{self.tp_rank}, in prepare_decode(), before create tensor input_ids={input_ids}, positions={positions}, slot_mapping={slot_mapping}, context_lens={context_lens}")

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(device=self.device, non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(device=self.device, non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(device=self.device, non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(device=self.device, non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        # print(f"{time.time()}, rank{self.tp_rank}, in prepare_decode(), after create tensor input_ids={input_ids}, {input_ids.size()}, positions={positions}, {positions.size()}, slot_mapping={slot_mapping}, {slot_mapping.size()}, context_lens={context_lens}, {context_lens.size()}, block_tables={block_tables}, {block_tables.size()}")

        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)

        # print(f"{time.time()}, rank{self.tp_rank}, in prepare_decode(), before return input_ids={input_ids}, {input_ids.size()}, positions={positions}, {positions.size()}, slot_mapping={slot_mapping}, {slot_mapping.size()}, context_lens={context_lens}, {context_lens.size()}, block_tables={block_tables}, {block_tables.size()}")
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_tensor: torch.Tensor, positions: torch.Tensor, is_prefill):
        # print(f"{time.time()}, rank{self.tp_rank}, in run_model() begin, is_prefill={is_prefill}, self.enforce_eager={self.enforce_eager}, nput_tensor.size={input_tensor.size()}")
        if is_prefill or self.enforce_eager or input_tensor.size(0) > 512:
            # print(f"{time.time()}, rank{self.tp_rank}, in run_model() before model run, input_tensor={input_tensor}, {input_tensor.size()}, positions={positions}, {positions.size()}")
            hidden_state = self.model(input_tensor, positions)
            # print(f"{time.time()}, rank{self.tp_rank}, in run_model() after model run, hidden_state={hidden_state}, {hidden_state.size()}")

            if self.pp_node_type == PPNodeType.PPNodeLast:
                logits = self.model.compute_logits(hidden_state)
                return logits
            else:
                hidden_state = hidden_state.squeeze(0)
                return hidden_state
        else:
            # print(f"{time.time()}, rank{self.tp_rank}, in run_model(), cuda-graph run path enter")
            bs = input_tensor.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()

            # print(
            #     f"{time.time()}, rank{self.tp_rank}, in run_model(), cuda-graph run path enter",
            #     f'input_tensor: {graph_vars["input_tensor"].size()}, {input_tensor.size()}, '
            #     f'positions: {graph_vars["positions"].size()}, {positions.size()}, '
            #     f'slot_mapping: {graph_vars["slot_mapping"].size()}, {context.slot_mapping.size()}, '
            #     f'context_lens: {graph_vars["context_lens"].size()}, {context.context_lens.size()}, '
            #     f'block_tables: {graph_vars["block_tables"].size()}, {context.block_tables.size()}, '
            # )

            graph_vars["input_tensor"][:bs] = input_tensor
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            # print(f'{time.time()}, rank{self.tp_rank}, in run_model() before model replay')
            graph.replay()
            torch.cuda.synchronize()
            # print(f'{time.time()}, rank{self.tp_rank}, in run_model() after model replay')

            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        assert self.pp_node_type == PPNodeType.PPNodeFirst
        
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)

        # print(f"{time.time()}, rank{self.tp_rank}, in run(), positions={positions}, {positions.device}, is_prefill={is_prefill}, input_ids={input_ids}")
        # print(f"{time.time()}, rank{self.tp_rank}, in run(), positions size={positions.size()}, input_ids size={input_ids.size()}")


        hidden_state = self.run_model(input_ids, positions, is_prefill)
        # print(f"{time.time()}, rank{self.tp_rank}, in run(), after run_model()")

        token_ids = None
        if self.tp_rank == 0:
            context = get_context()
            # print(f"{time.time()}, rank{self.tp_rank}, in run(), before send_pp_cmd(), context={context}")
            
            self.send_pp_cmd(context, hidden_state, positions)
            # print(f"{time.time()}, rank{self.tp_rank}, in run(), after send_pp_cmd()")

            temperatures = self.prepare_sample(seqs) if self.tp_rank == 0 else None
            logits = torch.empty([input_ids.size()[0], self.config.hf_config.vocab_size], dtype=torch.float16).cuda()

            # print(f"{time.time()}, rank{self.tp_rank}, in run(), before recv_pp_cmd()")
            self.recv_pp_cmd(logits)

            assert logits.dim() == 2


            # print(f"{time.time()}, rank{self.tp_rank}, in run(), after recv_pp_cmd(), logits={logits}, {logits.size()}, temperatures={temperatures}")
            token_ids = self.sampler(logits, temperatures).tolist() if self.tp_rank == 0 else None
            # print(f"{time.time()}, rank{self.tp_rank}, in run(), after sampler, token_ids={token_ids}")
        reset_context()

        # print(f"{time.time()}, rank{self.tp_rank}, in run(), before return, token_ids={token_ids}")
        return token_ids
    

    def run_non_first_node(self, hidden_state: torch.Tensor, positions: torch.Tensor, context: Context) -> list[int]:
        assert self.pp_node_type != PPNodeType.PPNodeFirst
        
        # print(f"{time.time()}, rank{self.tp_rank}, in run_non_first_node(), enter hidden_state={hidden_state}, positions={positions}")

        context.slot_mapping = context.slot_mapping.cuda(self.device)
        context.context_lens = context.context_lens.cuda(self.device)
        context.block_tables = context.block_tables.cuda(self.device)
        set_context(
            context.is_prefill,
            context.cu_seqlens_q,
            context.cu_seqlens_k,
            context.max_seqlen_q,
            context.max_seqlen_k,
            context.slot_mapping,
            context.context_lens,
            context.block_tables
        )

        hidden_state = hidden_state.cuda(self.device)
        positions = positions.cuda(self.device)

        # print(f"{time.time()}, rank{self.tp_rank}, in run_non_first_node(), before run_model()")
        hidden_state = self.run_model(hidden_state, positions, context.is_prefill)
        # print(f"{time.time()}, rank{self.tp_rank}, in run_non_first_node(), after run_model()")

        if self.tp_rank == 0:
            # print(f"{time.time()}, rank{self.tp_rank}, in run_non_first_node(), before send_pp_cmd() hidden_state={hidden_state}")
            self.send_pp_cmd(get_context(), hidden_state, positions)
            # print(f"{time.time()}, rank{self.tp_rank}, in run_non_first_node(), after send_pp_cmd()")

        
    


    @torch.inference_mode()
    def capture_cudagraph(self):
        get_rng_state = torch.cuda.get_rng_state
        set_rng_state = torch.cuda.set_rng_state
        rng_state = torch.cuda.get_rng_state()
        torch.cuda.get_rng_state = lambda: rng_state
        torch.cuda.set_rng_state = lambda _: None
    
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        if self.pp_node_type == PPNodeType.PPNodeFirst:
            input_tensor = torch.zeros(max_bs, dtype=torch.int64)
        else:
            input_tensor = torch.zeros([max_bs, hf_config.hidden_size])
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        if self.pp_node_type == PPNodeType.PPNodeLast:
            outputs = torch.zeros(max_bs, hf_config.vocab_size)
        else:
            outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1] # [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # print(f"{time.time()}, rank{self.tp_rank} in capture graph, positions={positions}, {positions.device}")

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_tensor[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                hidden_state = self.model(input_tensor[:bs], positions[:bs])    # capture
                outputs[:bs] = hidden_state
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_tensor=input_tensor,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

        torch.cuda.get_rng_state = get_rng_state
        torch.cuda.set_rng_state = set_rng_state
