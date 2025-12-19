from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import inspect
from src.instructions_registry import INSTRUCTION_DICT
import argparse

OAI_API_KEY = os.getenv("OAI_API_KEY")
OAI_BASE_URL = os.getenv("OAI_BASE_URL")

client_api = OpenAI(
    api_key=OAI_API_KEY,
    base_url=OAI_BASE_URL
)

rephrase_prompt = """
You are a helpful assistant that revise the prompt to test an LLMs' generalization ability.

The prompt typically contain one task and a few constraints. The only part that should be revised is the task description.

You are free to revise the task description but keep the new task compatible with the original constraints. Typically, the revised task should be more challenging than the original task.

You should make the minimal changes to the constraints unless absolutely necessary, such that the new prompt is still able to request the same set of constraints.

Both the responses to the original prompt and the new prompt will be evaluated by the following function:
{evaluation_function_implementation}

And the necessary arguments for the evaluation function (i.e., the requirements to be satisfied by the response) are:
{evaluation_function_arguments}

The original prompt is:
{prompt}

You should directly output the new prompt, without any other text.
""".strip()


def rephrase(prompt, evaluation_function_ids, evaluation_function_arguments, num_prompts=8):

    evaluation_function_implementations = """"""

    for instruction_id in evaluation_function_ids:
        instruction = INSTRUCTION_DICT[instruction_id]
        source_code = inspect.getsource(instruction)
        first_line = source_code.split("\n")[0] + f"\t# Corresponding to the constraint {instruction_id}"
        other_lines = source_code.split("\n")[1:]
        other_lines = "\n".join(other_lines)
        evaluation_function_implementations += first_line + "\n" + other_lines + "\n\n"

    evaluation_args = """"""
    for func_id, func_args in zip(evaluation_function_ids, evaluation_function_arguments):
        evaluation_args += f"- {func_id}: {func_args}\n"

    prompt = rephrase_prompt.format(
        prompt=prompt,
        evaluation_function_implementation=evaluation_function_implementations,
        evaluation_function_arguments=evaluation_args
    )

    new_prompts = []
    total_num = 0
    while total_num < num_prompts:
        batch_size = min(8, num_prompts - total_num)
        response = client_api.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            n=batch_size
        )
        total_num += batch_size
        new_prompts.extend([choice.message.content for choice in response.choices])
    return new_prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and target JSONL files.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the non-scalable input JSONL file")
    parser.add_argument("--target_path", type=str, required=True, help="Path to the target/output JSONL file")
    
    args = parser.parse_args()
    
    input_path = args.input_path
    target_path = args.target_path
    
    with open(input_path, "r") as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} prompts")
    
    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = [executor.submit(rephrase, data[i]["prompt"], data[i]["instruction_id_list"], data[i]["kwargs"], num_prompts=8) for i in range(len(data))]
        results = [future.result() for future in tqdm(futures)]
    print(f"Rephrased {len(data)} prompts")
    
    with open(target_path, "a") as f:
        for i in range(len(data)):
            for idx, rephrased_prompt in enumerate(results[i]):
                new_task = data[i].copy()
                new_task["key"] = f"{new_task['key']}:ct_alteration:{idx+1}"
                new_task["original_prompt"] = new_task["prompt"]
                new_task["prompt"] = rephrased_prompt
                f.write(json.dumps(new_task) + "\n")
    print(f"Saved prompts to {target_path}")