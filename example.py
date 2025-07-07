import os
import time
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
from argparse import ArgumentParser

try:
    from nanovllm.web_server import run_web_server_in_thread, model_req_queue, model_resp_queue
    from nanovllm import rpc
except:
    pass

prefill_tensor_dir = "/data/debug1"

def main(node_id: int, node_num: int):
    
    path = os.path.expanduser("/data/mmh/DeepSeek-V3-0324-AWQ")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=4, node_id=node_id, node_num=node_num)

    if node_id != 0:
        time.sleep(100000)
        raise SystemExit
    else:
        run_web_server_in_thread()

    sampling_params = SamplingParams(temperature=0, max_tokens=30)

    while True:
        
        prompts = [
            model_req_queue.get(),
        ]

        # call k100 without apply template
        remote_tensor_path = rpc.call_prefill(prompts[0])

        if remote_tensor_path is None:
            continue

        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            for prompt in prompts
        ]

        
        prefill_tensor_filename = os.path.basename(remote_tensor_path)
        prefill_tensor_filename = os.path.join(prefill_tensor_dir, prefill_tensor_filename)
        rpc.scp_copy(remote_tensor_path, prefill_tensor_filename)

        print("tensor copied, waiting nodes to load kvcache.")
        time.sleep(10)
        print(f"prompts={prompts}")
        outputs = llm.generate(prompts, sampling_params)

        model_resp_queue.put(None)
        llm.reset_engine()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--node-id", type=int, required=True)
    parser.add_argument("--node-num", type=int, required=True)
    cli_args = parser.parse_args()
    main(cli_args.node_id, cli_args.node_num)
