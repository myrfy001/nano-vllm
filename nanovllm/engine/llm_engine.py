import atexit
from dataclasses import fields
from time import perf_counter
import time
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch
from torch.profiler import ProfilerActivity

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fileds = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fileds}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.mp_queues = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            mp_queue = ctx.Queue()
            process = ctx.Process(target=ModelRunner, args=(config, i, mp_queue))
            process.start()
            self.ps.append(process)
            self.mp_queues.append(mp_queue)
            print(f"start process {i}")
        self.model_runner = ModelRunner(config, 0, self.mp_queues)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        print(f"{time.time()}, LLMEngine in step(), before model_runner.call, seqs={seqs}, is_prefill={is_prefill}")
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        print(f"{time.time()}, LLMEngine in step(), after model_runner.call")

        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        


        def core_inner(prof):
            prefill_throughput = decode_throughput = 0.
            prof_early_break_counter = 0
            while not self.is_finished():
                t = perf_counter()
                output, num_tokens = self.step()
                if use_tqdm:
                    if num_tokens > 0:
                        prefill_throughput = num_tokens / (perf_counter() - t)
                    else:
                        decode_throughput = -num_tokens / (perf_counter() - t)
                    pbar.set_postfix({
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    })
                for seq_id, token_ids in output:
                    outputs[seq_id] = token_ids
                    if use_tqdm:
                        pbar.update(1)
                
                
                if prof is not None:
                    prof.step()
                prof_early_break_counter += 1
                print(f"current step = {prof_early_break_counter}")
                # if prof_early_break_counter >= 30:
                #     break


        if True:
            with torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=10, active=300),
                record_shapes=True,
                with_stack=True,
                profile_memory=True,
            ) as prof:
                core_inner(prof)
                
            prof.export_chrome_trace(f"tracing.json.gz")
        else:
            core_inner(None)


        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
