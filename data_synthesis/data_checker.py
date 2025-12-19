from dotenv import load_dotenv
load_dotenv()
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import os
from pathlib import Path
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.instructions_registry import INSTRUCTION_DICT
import inspect
import argparse


OAI_API_KEY = os.getenv("OAI_API_KEY")
OAI_BASE_URL = os.getenv("OAI_BASE_URL")

client_api = OpenAI(
    api_key=OAI_API_KEY,
    base_url=OAI_BASE_URL
)


from pydantic import BaseModel


class AlignmentCheckResponse(BaseModel):
    reasoning: str
    is_valid: bool


prompt_template_function = """
You are tasked with verifying whether a given test case for evaluating LLMs' instruction-following behavior is valid.

Each test case consists of:
- **Prompt**: The instruction provided to the LLM.
- **Evaluation Arguments**: The specific parameters required by the evaluation function.
- **Evaluation Function**: The function that checks the LLM's response against the evaluation arguments and returns a boolean judgment.

Your task is to determine if the test case is valid by checking the following conditions:

A test case is valid only if it satisfies all three conditions below:

1. **Argument Reflection**  
   Every evaluation argument must be explicitly required or constrained in the Prompt.  
   If the Prompt does not impose a constraint corresponding to a given evaluation argument, the test case is invalid.
   You should understand how the evaluation function works and the constraints (evaluation arguments) to be evaluated.

2. **Format Compatibility**  
   The Prompt's format and placeholder conventions must allow a response that can pass the evaluation function.  
   - For arguments of type `keywords`, `keyword`, `forbidden_words`, or `prompt_to_repeat`, the Prompt must clearly require or forbid these elements in such a way that the evaluation function can check them.
   - Be careful with the punctuation marks.

3. **Logical Consistency**  
   Logically, the context in the Prompt (e.g., the tasks or other constraints that are not evaluated) must not contradict the constraints to be evaluated.
   All constraints are independent of each other and should be met at the same time. Do not interpret one constraint conditioned on another.
   For example, requiring JSON-formatted output but additionally asking for a plain-text line is invalid.

### Notes:
- Your judgement should be harsh and rigorous. You should not be lenient or allow for any exceptions. Default to "is_valid": false.
- A stricter or looser Prompt constraint is **not** equivalent to an Evaluation Argument.
- The test case is valid only if **all three conditions** above are satisfied.

### Output:
Return your judgment in the following JSON format:
```json
{{
    "reasoning": "Your reasoning process",
    "is_valid": true/false
}}
```

⸻

Prompt:
{prompt}

Evaluation Arguments:
{evaluation_args}

Evaluation Function:
{evaluation_function}
""".strip()


def alignment_check(json_data):
    prompt = json_data["prompt"]
    evaluation_function_ids = json_data["instruction_id_list"]
    evaluation_function_arguments = json_data["kwargs"]
    evaluation_function_implementations = """"""
    
    try:
        evaluation_function_implementations = ""

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

        prompt_diff = prompt_template_function.format(
            prompt=prompt,
            evaluation_args=evaluation_args,
            evaluation_function=evaluation_function_implementations
        )
        try:

            response = client_api.beta.chat.completions.parse(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt_diff}],
                temperature=0.00000001,
                n=1,
                response_format=AlignmentCheckResponse,
                timeout=30,
                reasoning_effort="medium"
            )
            check_results = {
                "reasoning": response.choices[0].message.parsed.reasoning,
                "is_valid": response.choices[0].message.parsed.is_valid,
            }
        except Exception as e:
            print(f"Error in alignment_check: {e}")
            check_results = {
                "reasoning": f"Error in alignment_check: {e}",
                "is_valid": False,
            }
    except Exception as e:
        print(f"Error in processing instruction functions: {e}")
        check_results = {
            "reasoning": f"Error in processing instruction functions: {e}",
            "is_valid": False,
        }
        
    json_data["validity-checking"] = check_results
    return json_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input file and generate annotation paths")
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to the input JSONL file"
    )
    parser.add_argument(
        "--target_dir", 
        type=str, 
        required=True, 
        help="Directory to save annotated and filtered files"
    )
    
    args = parser.parse_args()
    
    input_file = args.input_file
    target_dir = args.target_dir
    
    file_name = Path(input_file).name
    annotation_path = Path(target_dir) / f"{file_name.split('.')[0]}_annotated.jsonl"
    filtered_path = Path(target_dir) / f"{file_name.split('.')[0]}_filtered.jsonl"
    
    with open(annotation_path, "w") as f:
        pass
    with open(filtered_path, "w") as f:
        pass
    
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    datas = []
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            datas.append(data)
    print(f"Loaded {len(datas)} prompts")
    
    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = [executor.submit(alignment_check, json_data) for json_data in datas]
        results = [future.result() for future in tqdm(futures)]
    
    from collections import Counter
    validity_counter = Counter()

    for result in results:
        with open(annotation_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        try:
            root_key = result["key"].split(":")[0]
        except Exception as e:
            root_key = result["key"]
        if root_key not in validity_counter:
            validity_counter[root_key] = 0
        
        if "Error in processing instruction functions: " in result["validity-checking"]["reasoning"] or "Error in alignment_check: " in result["validity-checking"]["reasoning"]:
            continue

        if result["validity-checking"]["is_valid"]:
            validity_counter[root_key] += 1
            with open(filtered_path, "a") as f:
                f.write(json.dumps(result) + "\n")
                
    print(f"Saved to {annotation_path} and {filtered_path}")