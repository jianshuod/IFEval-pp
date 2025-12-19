import copy
import json
from tqdm import tqdm
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


from dotenv import load_dotenv
load_dotenv()

from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
from openai import OpenAI
from string import Template
import immutabledict
import inspect
from src.instructions_registry import INSTRUCTION_DICT
import argparse


OAI_API_KEY = os.getenv("OAI_API_KEY")
OAI_BASE_URL = os.getenv("OAI_BASE_URL")

client_api = OpenAI(
    api_key=OAI_API_KEY,
    base_url=OAI_BASE_URL
)

from pydantic import BaseModel

class ReviseFillinResponse(BaseModel):
    random_seed: str
    reasoning: str
    revised_prompt: str
    revised_instruction_id_list: list[str]
    revised_kwargs: str

class AllReviseFillinResponse(BaseModel):
    outputs: list[ReviseFillinResponse]

auto_fill_prompt = """
You are a helpful assistant that revise the prompt to test an LLMs' generalization ability.

The prompt typically contain one task and a few constraints. You are allowed to revise the constraints and correspondingly adjust the evaluation arguments. If necessary, you can also revise the task description.

If multiple constraints exist, you should revise at least one of them.

Typically, the revised constraints should be **more challenging** than the original constraints, but not beyond the **allowable range** of the evaluation functions.

Keep using the original constraint types of the original prompt and the original evaluation functions. Only change the fill-in values (arguments) of the constraints.

To help you better understand the constraints, here are the evaluation functions that will be used to evaluate the responses to the original prompt and the new prompt:
{evaluation_function_implementations}

The original prompt is:
{prompt}

The original instruction id list is:
{instruction_id_list}

The evaluation arguments for the original prompt are:
{evaluation_function_arguments}

You should output a list of JSON objects with the following fields:
- "random_seed": The random seed you choose to generate the new prompt.
- "reasoning": The reasoning process for generating the new prompt and the new kwargs.
- "revised_prompt": The new prompt.
- "revised_instruction_id_list": The new instruction id list. If you have added or removed some constraints, you should also update the instruction id list. The length of the list should be the same as that of the revised kwargs.
- "revised_kwargs": The new kwargs in string format, which should be valid JSON. The format should be the same as the original kwargs as follows:
{evaluation_function_arguments_format}

The total number of the revised prompts should be {num_prompts}. The revised prompts should be **distinct** in constraints, **reasonable**, and increasingly difficult.

Keep in mind that do not use constraints beyond the provided evaluation functions.
""".strip()



def alter_fillin(prompt, evaluation_function_ids, evaluation_function_arguments, num_prompts=4):
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

    instruction_id_list_str = json.dumps(evaluation_function_ids, indent=0)
    evaluation_function_arguments_format = json.dumps(evaluation_function_arguments, indent=0)

    prompt = auto_fill_prompt.format(
        prompt=prompt,
        instruction_id_list=instruction_id_list_str,
        evaluation_function_implementations=evaluation_function_implementations,
        evaluation_function_arguments=evaluation_args,
        evaluation_function_arguments_format=evaluation_function_arguments_format,
        num_prompts=num_prompts
    )

    while True:
        response = client_api.beta.chat.completions.parse(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            n=1,
            response_format=AllReviseFillinResponse
        )
        try:
            new_prompts = [(response.revised_prompt, response.revised_instruction_id_list, response.revised_kwargs) for response in response.choices[0].message.parsed.outputs]
            revised_kwargs = [json.loads(kwarg_str) for _, _, kwarg_str in new_prompts]
            break
        except Exception as e:
            print(e)
            continue

    return new_prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input and target JSONL files.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the scalable input JSONL file")
    parser.add_argument("--target_path", type=str, required=True, help="Path to the target/output JSONL file")
    
    args = parser.parse_args()
    
    input_path = args.input_path
    target_path = args.target_path
    
    with open(input_path, "r") as f:
        datas = [json.loads(line) for line in f]
    print(f"Loaded {len(datas)} prompts")

    def _run(data):
        return alter_fillin(
            data["prompt"], data["instruction_id_list"], data["kwargs"], num_prompts=8
        )

    with ThreadPoolExecutor(max_workers=128) as executor:
        all_results = list(tqdm(executor.map(_run, datas), total=len(datas)))

    with open(target_path, "a") as f:
        for i in range(len(datas)):
            for idx, (revised_prompt, revised_instruction_id_list, revised_kwargs) in enumerate(all_results[i]):
                new_task = datas[i].copy()
                new_task["key"] = f"{new_task['key']}:ct_alteration:{idx+1}"
                new_task["original_prompt"] = new_task["prompt"]
                new_task["original_instruction_id_list"] = new_task["instruction_id_list"]
                new_task["original_kwargs"] = new_task["kwargs"]
                new_task["prompt"] = revised_prompt
                new_task["instruction_id_list"] = revised_instruction_id_list
                new_task["kwargs"] = json.loads(revised_kwargs)
                f.write(json.dumps(new_task) + "\n")
    print(f"Save to {target_path}, total {len(all_results)} cases.")