import pickle
import time
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config, PPNodeType
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.deepseek_v3 import DeepseekV3ForCausalLLMFirst, DeepseekV3ForCausalLLMMiddle, DeepseekV3ForCausalLLMLast
                                        
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context, Context, set_tp_context_info
from nanovllm.utils.loader import load_model
from nanovllm.utils.serdes import invoke_deserialize_context_kernel, invoke_serialize_context_kernel

from safetensors.torch import save_file

class ModelRunner:

    def __init__(self, config: Config, tp_rank: int, mp_queue):
        
        config.local_rank = tp_rank

        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.tp_world_size = config.tensor_parallel_size
        self.world_size = config.node_num * config.tensor_parallel_size
        self.rank = config.node_id * config.tensor_parallel_size + tp_rank
        self.tp_rank = tp_rank
        self.mp_queue = mp_queue
        self.device = torch.device("cuda", self.tp_rank)

        self.buf_tensor = torch.empty((32, 7168*2), dtype=torch.uint8, device=self.device)
        self.meta_buf = torch.empty(8, dtype=torch.uint16, device=self.device)
        self.slot_mapping_buf = torch.empty([2048,], dtype=torch.int32, device=self.device)
        self.context_lens_buf = torch.empty([2048,], dtype=torch.int32, device=self.device)
        self.block_tables_buf = torch.empty([1, 2048,], dtype=torch.int32, device=self.device)
        self.hidden_state_buf = torch.empty([128, self.config.hf_config.hidden_size], dtype=torch.float16, device=self.device)
        self.positions_buf = torch.empty([1024,], dtype=torch.int64, device=self.device)

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
        pp_end_layer_id -= 1
        self.pp_start_layer_id = pp_start_layer_id


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
        self.model.load_weight(config.model, start_layer=pp_start_layer_id, end_layer=pp_end_layer_id)
        self.pp_node_type = pp_node_type

        self.sampler = Sampler()
        self.allocate_kv_cache(config.gpu_memory_utilization)

        print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in ModelRunner __init__ before global barrier", flush=True)
        dist.barrier()
        print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in ModelRunner __init__ after global barrier", flush=True)

        # import pdb; pdb.set_trace()



        if not self.enforce_eager:
            self.capture_cudagraph()

        # torch.set_default_device("cpu")
        # torch.set_default_dtype(default_dtype)

        
        if self.tp_world_size > 1:
            if tp_rank == 0:
                try:
                    self.shm = SharedMemory(name="nanovllm-1", create=True, size=10*2**20)
                except FileExistsError:
                    self.shm = SharedMemory(name="nanovllm-1", size=10*2**20)

                dist.barrier(group=self.tp_group)
                
                # only the first node's first process doesn't run the loop.
                if(pp_node_type != PPNodeType.PPNodeFirst):
                    self.loop()
            else:
                dist.barrier(group=self.tp_group)
                self.shm = SharedMemory(name="nanovllm-1")
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
        if False:
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
            
            
            self.recv_pp_cmd()

            print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in loop(), get new pp cmd", flush=True)

            args = (self.buf_tensor,)
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

            invoke_serialize_context_kernel(
                context.is_prefill, context.slot_mapping, context.context_lens, context.block_tables, hidden_state, positions, self.buf_tensor
            )

            # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, dist.send size={self.buf_tensor.size()}, dtype={self.buf_tensor.dtype} self.buf_tensor", flush=True)
            torch.cuda.synchronize()
            dist.send(self.buf_tensor, self.next_pp_head_node_global_rank)
            if self.tp_rank == 0:
                save_file({"buf_tensor":self.buf_tensor}, "dumps/buf_tensor_send.safetensor")
                # raise SystemExit
            torch.cuda.synchronize()
        else:
            # for last node, send logits instead of hidden state
            # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, dist.send size={hidden_state.size()}, dtype={hidden_state.dtype} logits", flush=True)
            torch.cuda.synchronize()
            dist.send(hidden_state, self.next_pp_head_node_global_rank)
            torch.cuda.synchronize()

    def recv_pp_cmd(self, logit_tensor=None):
        assert self.tp_rank == 0

        if self.pp_node_type != PPNodeType.PPNodeFirst:
            # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, dist.recv size={self.buf_tensor.size()}, dtype={self.buf_tensor.dtype} self.buf_tensor", flush=True)
            dist.recv(self.buf_tensor, self.prev_pp_head_node_global_rank)
            torch.cuda.synchronize()
        else:
            assert logit_tensor is not None
            # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, dist.recv size={logit_tensor.size()}, dtype={logit_tensor.dtype} logit_tensor", flush=True)
            dist.recv(logit_tensor, self.prev_pp_head_node_global_rank)
            torch.cuda.synchronize()
            
            


    def read_shm(self):
        assert self.tp_world_size > 1 and self.tp_rank
        method_name, args = self.mp_queue.get()
        # n = int.from_bytes(self.shm.buf[0:4], "little")
        # method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        # self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.tp_world_size > 1 and not self.tp_rank
        # data = pickle.dumps([method_name, *args])
        # n = len(data)
        # print(f"n={n}, {self.shm.size}", flush=True)
        # assert n + 4 <= self.shm.size
        # self.shm.buf[0:4] = n.to_bytes(4, "little")
        # self.shm.buf[4:n+4] = data
        for mp_queue in self.mp_queue:
            mp_queue.put((method_name, args))

    

    def call(self, method_name, *args):
        if self.tp_world_size > 1 and self.tp_rank == 0:
            if method_name == "run":
                self.write_shm(method_name, *args)
            elif method_name == "run_non_first_node":
                self.write_shm(method_name)
                # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, send serialized data by broadcast, src={self.rank}", flush=True)
                dist.broadcast(self.buf_tensor, src=self.rank, group=self.tp_group)
                # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, finish send serialized data by broadcast, src={self.rank}", flush=True)
            else:
                raise Exception("Not supported function call")
            
        if self.tp_rank > 0:
            if method_name == "run_non_first_node":
                # recv serialized data
                # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, recv serialized data by broadcast, src={self.rank-self.tp_rank}", flush=True)
                dist.broadcast(self.buf_tensor, src=self.rank-self.tp_rank, group=self.tp_group)
                # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, after recv serialized data by broadcast, src={self.rank-self.tp_rank}", flush=True)
                args = (self.buf_tensor,)

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
        self.kv_cache = torch.zeros(hf_config.num_hidden_layers, min(4096, config.num_kvcache_blocks), self.block_size, num_kv_heads, kv_cache_head_dim)
        # self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        
        from safetensors import safe_open
        with safe_open("small_phase0_seqlen4_start0.safetensor", "pt", "cpu") as safetensor_file:
            loaded_kv_cache = safetensor_file.get_tensor("out_kvcache")
            self.kv_cache[:,:4] = loaded_kv_cache.to(self.device)

        if self.rank == 0:
            print(f"loaded kv cache = {loaded_kv_cache}")
            print(f"self.kv_cache for first layer = {self.kv_cache[0]}")
            

        layer_id = self.pp_start_layer_id
        for module in self.model.modules():
            # if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            #     module.k_cache = self.kv_cache[0, layer_id]
            #     module.v_cache = self.kv_cache[1, layer_id]
            #     layer_id += 1
            if hasattr(module, "kv_c_and_k_pe_cache"):
                module.kv_c_and_k_pe_cache = self.kv_cache[layer_id]
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

    @torch.inference_mode()
    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []

        
        
        for seq in seqs:
            print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in prepare_decode(), {seq=}")
            input_ids.append(seq.last_token)
            positions.append(len(seq)-1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        
        # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in prepare_decode(), before create tensor input_ids={input_ids}, positions={positions}, slot_mapping={slot_mapping}, context_lens={context_lens}")

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(device=self.device, non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(device=self.device, non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(device=self.device, non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(device=self.device, non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)

        print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in prepare_decode(), after create tensor input_ids={input_ids}, {input_ids.size()}, positions={positions}, {positions.size()}, slot_mapping={slot_mapping}, {slot_mapping.size()}, context_lens={context_lens}, {context_lens.size()}, block_tables={block_tables}, {block_tables.size()}")

        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)

        self.graph_vars["slot_mapping"][0] = slot_mapping[0]
        self.graph_vars["context_lens"][0] = context_lens[0]
        self.graph_vars["block_tables"][0][:block_tables.size(1)] = block_tables
        self.graph_vars["input_tensor"][0] = input_ids[0]
        self.graph_vars["positions"][0] = positions[0]

        # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in prepare_decode(), before return input_ids={input_ids}, {input_ids.size()}, positions={positions}, {positions.size()}, slot_mapping={slot_mapping}, {slot_mapping.size()}, context_lens={context_lens}, {context_lens.size()}, block_tables={block_tables}, {block_tables.size()}")
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_tensor: torch.Tensor, positions: torch.Tensor, is_prefill):
        # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run_model() begin, is_prefill={is_prefill}, self.enforce_eager={self.enforce_eager}, nput_tensor.size={input_tensor.size()}", flush=True)
        if is_prefill or self.enforce_eager or input_tensor.size(0) > 512:
            # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run_model() before model run, input_tensor={input_tensor}, {input_tensor.size()}, positions={positions}, {positions.size()}")
            hidden_state = self.model(input_tensor, positions)
            # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run_model() after model run, hidden_state={hidden_state}, {hidden_state.size()}")

            if self.pp_node_type == PPNodeType.PPNodeLast:
                logits = self.model.compute_logits(hidden_state)
                return logits
            else:
                hidden_state = hidden_state.squeeze(0)
                return hidden_state
        else:
            # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run_model(), cuda-graph run path enter")
            bs = 1 # TODO: FIXME 
            # bs = input_tensor.size(0)
            
            # graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # for k, v in graph_vars.items():
            #     if k != "outputs":
            #         v.zero_()

            # print(
            #     f"{time.time()}, rank{self.tp_rank},{self.rank}, in run_model(), cuda-graph run path enter",
            #     f'input_tensor: {graph_vars["input_tensor"].size()}, {input_tensor.size()}, '
            #     f'positions: {graph_vars["positions"].size()}, {positions.size()}, '
            #     f'slot_mapping: {graph_vars["slot_mapping"].size()}, {context.slot_mapping.size()}, '
            #     f'context_lens: {graph_vars["context_lens"].size()}, {context.context_lens.size()}, '
            #     f'block_tables: {graph_vars["block_tables"].size()}, {context.block_tables.size()}, '
            # )

            # if self.pp_node_type == PPNodeType.PPNodeFirst:
            #     context = get_context()
            #     graph_vars["input_tensor"][:bs] = input_tensor
            #     graph_vars["positions"][:bs] = positions
            #     graph_vars["slot_mapping"][:bs] = context.slot_mapping
            #     graph_vars["context_lens"][:bs] = context.context_lens
            #     graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            if self.rank == 0:
                # print(f'{time.time()}, rank{self.tp_rank},{self.rank}, in run_model() before model replay, graph_vars={graph_vars}', flush=True)
                print(f'{time.time()}, rank{self.tp_rank},{self.rank}, in run_model() before model replay, kvcache={self.kv_cache[1]}', flush=True)

            # graph.replay()
            
            graph_vars["outputs"][:bs] = self.model(graph_vars["input_tensor"][:bs], graph_vars["positions"][:bs])

            if self.pp_node_type == PPNodeType.PPNodeFirst:
                save_file({"kv_cache":self.kv_cache}, f"dumps/{time.time()}_rank{self.rank}-all_kv_caches.safetensor")

            torch.cuda.synchronize()
            # print(f'{time.time()}, rank{self.tp_rank},{self.rank}, in run_model() after model replay', flush=True)

            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        assert self.pp_node_type == PPNodeType.PPNodeFirst
        
        if is_prefill:
            # input_ids, positions = self.prepare_prefill(seqs)

            from safetensors import safe_open
            with safe_open("small_phase0_seqlen4_start0.safetensor", "pt", "cpu") as safetensor_file:
                loaded_logit = safetensor_file.get_tensor("out_logits")
                loaded_token_ids = safetensor_file.get_tensor("tokens")
                token_id = loaded_logit.argmax(dim=-1)

            print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run() prefill before modify seq, seqs[0].token_ids={seqs[0].token_ids}, num_tokens={seqs[0].num_tokens}, num_prompt_tokens={seqs[0].num_prompt_tokens}, num_cached_tokens={seqs[0].num_cached_tokens}, block_table={seqs[0].block_table}")

            seqs[0].token_ids = loaded_token_ids[0].tolist()
            seqs[0].num_tokens = len(seqs[0].token_ids)
            seqs[0].num_prompt_tokens = len(seqs[0].token_ids)
            seqs[0].num_cached_tokens = len(seqs[0].token_ids)
            seqs[0].block_table = [i for i in range(len(seqs[0].token_ids))]

            print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run() prefill after modify seq, seqs[0].token_ids={seqs[0].token_ids}")
            
            token_ids = [int(token_id[0])]
            return token_ids
        else:
            input_ids, positions = self.prepare_decode(seqs)


        # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run(), positions={positions}, {positions.device}, is_prefill={is_prefill}, input_ids={input_ids}")
        # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run(), positions size={positions.size()}, input_ids size={input_ids.size()}")


        hidden_state = self.run_model(input_ids, positions, is_prefill)
        # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run(), after run_model()")

        token_ids = None
        if self.tp_rank == 0:
            context = get_context()
            print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run(), before send_pp_cmd(), context={context}")
            
            self.send_pp_cmd(context, hidden_state, positions)
            # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run(), after send_pp_cmd()")

            temperatures = self.prepare_sample(seqs) if self.tp_rank == 0 else None
            logits = torch.empty([input_ids.size()[0], self.config.hf_config.vocab_size], dtype=torch.float16).cuda()

            # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run(), before recv_pp_cmd()")
            self.recv_pp_cmd(logits)

            assert logits.dim() == 2


            # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run(), after recv_pp_cmd(), logits={logits}, {logits.size()}, temperatures={temperatures}")
            token_ids = self.sampler(logits, temperatures).tolist() if self.tp_rank == 0 else None
            # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run(), after sampler, token_ids={token_ids}")
        reset_context()

        # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run(), before return, token_ids={token_ids}")
        return token_ids
    

    def run_non_first_node(self, buf_tensor: torch.Tensor) -> list[int]:
        assert self.pp_node_type != PPNodeType.PPNodeFirst
        
        # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run_non_first_node(), enter hidden_state={hidden_state}, positions={positions}")
        # local_buf_tensor = buf_tensor.to(self.device, non_blocking=True)

        print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run_non_first_node(), before deserialize, {buf_tensor=}")
        print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run_non_first_node(), before deserialize, {self.graph_vars=}")

        # if self.tp_rank == 0:
        #     save_file({"buf_tensor":buf_tensor}, "dumps/buf_tensor_recv.safetensor")
        #     raise SystemExit

        invoke_deserialize_context_kernel(
            buf_tensor,
            self.meta_buf,
            self.graph_vars["slot_mapping"],
            self.graph_vars["context_lens"],
            self.graph_vars["block_tables"],
            self.graph_vars["input_tensor"][:1],
            self.graph_vars["positions"][:1]
        )
        
        torch.cuda.synchronize()
        local_meta_buf_slice = self.meta_buf[0:6].tolist()

        print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run_non_first_node(), after deserialize, {self.graph_vars=}")
        
        print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run_non_first_node(), local_meta_buf_slice={local_meta_buf_slice}", flush=True)
        is_prefill = bool(int(local_meta_buf_slice[0]))
        slot_mapping_size = local_meta_buf_slice[1]
        context_lens_size = local_meta_buf_slice[2]
        block_tables_size = local_meta_buf_slice[3]
        hidden_state_size = local_meta_buf_slice[4]
        positions_size    = local_meta_buf_slice[5]

        slot_mapping = self.graph_vars["slot_mapping"][:slot_mapping_size]
        context_lens = self.graph_vars["context_lens"][:context_lens_size]
        block_tables = self.graph_vars["block_tables"][:1][:block_tables_size]
        hidden_state = self.graph_vars["input_tensor"][:hidden_state_size]
        positions = self.graph_vars["positions"][:positions_size]


        print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run_non_first_node(), before set context, {slot_mapping=}, {context_lens=}, {block_tables=}", flush=True)

        # TODO: FIXME: hard code to None and 0 since not support batch and cumulate now
        set_context(
            is_prefill,
            None,
            None,
            0,
            0,
            slot_mapping,
            context_lens,
            block_tables
        )


        print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run_non_first_node(), before run_model(), context={get_context()}", flush=True)
        hidden_state = self.run_model(hidden_state, positions, is_prefill)
        # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run_non_first_node(), after run_model()")

        if self.tp_rank == 0:
            # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run_non_first_node(), before send_pp_cmd() hidden_state={hidden_state}")
            self.send_pp_cmd(get_context(), hidden_state, positions)
            # print(f"{time.time()}, rank{self.tp_rank},{self.rank}, in run_non_first_node(), after send_pp_cmd()")

        
    


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

        # for bs in reversed(self.graph_bs):
        #     graph = torch.cuda.CUDAGraph()
        #     set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
        #     outputs[:bs] = self.model(input_tensor[:bs], positions[:bs])    # warmup
        #     with torch.cuda.graph(graph, self.graph_pool):
        #         hidden_state = self.model(input_tensor[:bs], positions[:bs])    # capture
        #         outputs[:bs] = hidden_state
        #     if self.graph_pool is None:
        #         self.graph_pool = graph.pool()
        #     self.graphs[bs] = graph
        #     torch.cuda.synchronize()
        #     reset_context()

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
