import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
from argparse import ArgumentParser


def main(node_id: int):
    path = os.path.expanduser("/data/mmh/DeepSeek-V3-0324-AWQ")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=4, node_id=node_id)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
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
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--node-id", type=int, required=True)
    cli_args = parser.parse_args()
    main(cli_args.node_id)
