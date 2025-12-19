from dotenv import load_dotenv
load_dotenv()

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from openai import OpenAI
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

add_distractor_prompt = """
You are a helpful assistant tasked with adding one additional distractor constraint to the original prompt.

### Instructions:
- Preserve all contents of the original prompt exactly as they are.  
- At the end of the new prompt, append one extra constraint sentence.  
- This extra constraint must introduce an additional **format requirement** for the response.  
- The new constraint must not interfere with or alter the original constraints.  

### Original Constraints:
{constraints}

### Evaluation Functions:
{evaluation_functions}

### Original Prompt:
{prompt}
""".strip()

def add_distractor(prompt, evaluation_function_ids, evaluation_function_arguments, num_prompts=8):

    evaluation_function_implementations = ""

    for instruction_id in evaluation_function_ids:
        instruction = INSTRUCTION_DICT[instruction_id]
        source_code = inspect.getsource(instruction)
        first_line = source_code.split("\n")[0] + f"\t# Corresponding to the constraint {instruction_id}"
        other_lines = source_code.split("\n")[1:]
        other_lines = "\n".join(other_lines)
        evaluation_function_implementations += first_line + "\n" + other_lines + "\n\n"


    evaluation_args = ""
    for func_id, func_args in zip(evaluation_function_ids, evaluation_function_arguments):
        evaluation_args += f"- {func_id}: {func_args}\n"

    prompt = add_distractor_prompt.format(
        prompt=prompt,
        constraints=evaluation_args,
        evaluation_functions=evaluation_function_implementations
    )

    new_prompts = []
    current_num = 0
    while current_num < num_prompts:
        batch_size = min(8, num_prompts - current_num)
        response = client_api.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            n=batch_size
        )
        new_prompts.extend([choice.message.content for choice in response.choices])
        current_num += batch_size
    return new_prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and output JSONL files.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file")
    
    args = parser.parse_args()
    
    input_file = args.input_file
    output_file = args.output_file
    
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} prompts")
    
    with ThreadPoolExecutor(max_workers=256) as executor:
        futures = [executor.submit(add_distractor, data[i]["prompt"], data[i]["instruction_id_list"], data[i]["kwargs"], num_prompts=8) for i in range(len(data))]
        results = [future.result() for future in tqdm(futures)]
    print(f"Added distractor to {len(data)} prompts")
    
    with open(output_file, "w") as f:
        for i in range(len(data)):
            for idx, distractor_prompt in enumerate(results[i]):
                new_task = data[i].copy()
                new_task["key"] = f"{new_task['key']}:distractor:{idx + 1}"
                new_task["original_prompt"] = new_task["prompt"]
                new_task["prompt"] = distractor_prompt
                f.write(json.dumps(new_task) + "\n")
    print(f"Saved {len(data)} prompts to {output_file}")