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

rephrase_prompt = """
You are a helpful assistant whose task is to rephrase a given prompt into another prompt.

You will be provided with:
1. A complete natural language prompt consisting of a task and one or more constraints.
2. A set of evaluation functions that will be used to evaluate responses to the prompt.
3. The arguments for the evaluation functions.

### Your task
Rewrite the given prompt while following these rules:
- Treat the input as a single, complete prompt. Keep all constraints intact, but rephrase them naturally.
- Do not alter numbers, thresholds, links, counts, or formatting requirements.
- Do not add new constraints or information.
- Preserve explicit structural rules (e.g., “repeat exactly,” “use this format,” “separate items with ******”).
- Always output the rewritten prompt in English, even if the original is in another language.
- If a constraint specifies a casing requirement (e.g., ALL CAPS, all lowercase), rephrase it using normal sentence casing.
- For constraints in the **repeat_prompt** category, do not modify the sentence to be repeated (the one provided in the arguments).

---

### Input
**Original prompt:**
{prompt}

**Evaluation functions:**
{evaluation_function_implementation}

**Arguments:**
{evaluation_function_arguments}

---

### Output
Directly produce the rewritten prompt, without any additional text.
""".strip()


def rephrase(prompt, evaluation_function_ids, evaluation_function_arguments, num_prompts=8):

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

    prompt = rephrase_prompt.format(
        prompt=prompt,
        evaluation_function_implementation=evaluation_function_implementations,
        evaluation_function_arguments=evaluation_args
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
    parser = argparse.ArgumentParser(description="Process input and output files.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--target_file", type=str, required=True, help="Path to the output JSONL file")
    
    args = parser.parse_args()
    
    input_file = args.input_file
    target_file = args.target_file
    
    with open(input_file, "r") as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} prompts")
    
    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = [executor.submit(rephrase, data[i]["prompt"], data[i]["instruction_id_list"], data[i]["kwargs"], num_prompts=24) for i in range(len(data))]
        results = [future.result() for future in tqdm(futures)]
    print(f"Rephrased {len(data)} prompts")
    with open(target_file, "a") as f:
        for i in range(len(data)):
            for idx, rephrased_prompt in enumerate(results[i]):
                new_task = data[i].copy()
                new_task["key"] = f"{new_task['key']}:rephrasing:{idx + 1}"
                new_task["original_prompt"] = new_task["prompt"]
                new_task["prompt"] = rephrased_prompt
                f.write(json.dumps(new_task) + "\n")
    print(f"Saved {len(data)} prompts to {target_file}")