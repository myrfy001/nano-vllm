import os
import time
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
from argparse import ArgumentParser


def main(node_id: int, node_num: int):
    path = os.path.expanduser("/data/mmh/DeepSeek-V3-0324-AWQ")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=4, node_id=node_id, node_num=node_num)

    if node_id != 0:
        time.sleep(100000)
        raise SystemExit

    sampling_params = SamplingParams(temperature=0, max_tokens=30)
    prompts = [
        "你是谁",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]

    print(f"prompts={prompts}")
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--node-id", type=int, required=True)
    parser.add_argument("--node-num", type=int, required=True)
    cli_args = parser.parse_args()
    main(cli_args.node_id, cli_args.node_num)
