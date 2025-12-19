from dotenv import load_dotenv
load_dotenv()
import json
import os
from pathlib import Path
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading
from tqdm import tqdm
from openai import OpenAI

from datasets import load_dataset
ds = load_dataset("RebeccaYU920/ifeval-pp")

parser = ArgumentParser()
parser.add_argument("--model", type=str, default="gpt-4.1")
parser.add_argument("--client_type", type=str, default="vllm", choices=["vllm", "oai"])
parser.add_argument("--input_data", type=str, default="assets/ifeval_original.jsonl")
parser.add_argument("--output_dir", type=str, default="results/original/responses")
parser.add_argument("--runname", type=str, default=None)
parser.add_argument("--num_workers", type=int, default=16)
parser.add_argument("--port", type=int, default=8102)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--system_prompt", type=str, default=None)
parser.add_argument("--reasoning-effort", type=str, default=None, choices=["low", "medium", "high", "minimal", None])
args = parser.parse_args()

# if -no-think
if "-no-think" in args.model:
    model_name = args.model.replace("-no-think", "")
    no_think = True
else:
    model_name = args.model
    no_think = False
print(f"Model: {args.model}")
client_type = args.client_type

if not args.runname:
    runname = args.model
else:
    runname = args.runname

input_jsonl_file = args.input_data
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
output_jsonl_file = output_dir / f"{runname}.jsonl"

# ---- Thread-local client factory ----
_tls = threading.local()

def get_client():
    if getattr(_tls, "client", None) is not None:
        return _tls.client

    if client_type == "vllm":
        base_url = os.getenv("VLLM_BASE_URL", f"http://localhost:{args.port}/v1")
        api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        _tls.client = OpenAI(api_key=api_key, base_url=base_url)
    else:
        base_url = os.getenv("OAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OAI_API_KEY") or ""
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY/OAI_API_KEY for client_type='oai'.")
        _tls.client = OpenAI(api_key=api_key, base_url=base_url)
    return _tls.client

# ---- Load all prompts ----
with open(input_jsonl_file, "r", encoding="utf-8") as f:
    queries = [json.loads(line.strip()) for line in f if line.strip()]
print(f"Total queries loaded: {len(queries)}")

# ---- Load completed keys ----
completed_keys = set()
if output_jsonl_file.exists():
    with open(output_jsonl_file, "r", encoding="utf-8") as f_out:
        for line in f_out:
            try:
                obj = json.loads(line.strip())
                if "key" in obj:
                    completed_keys.add(obj["key"])
            except:
                continue
print(f"Already completed: {len(completed_keys)}")

# ---- Worker function ----
def process_prompt(query, lock):
    key = query.get("key", "")
    if key in completed_keys:
        return None  

    user_prompt = query.get("prompt", "")
    try:
        client = get_client()
        additional_kwargs = {}
        if no_think:
            additional_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        if args.system_prompt:
            messages = [{"role": "system", "content": args.system_prompt}, {"role": "user", "content": user_prompt}]
        else:
            messages = [{"role": "user", "content": user_prompt}]
        
        if args.reasoning_effort:
            additional_kwargs["reasoning_effort"] = args.reasoning_effort

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=args.temperature,
            **additional_kwargs
        )
        response = (chat_completion.choices[0].message.content or "").strip()
    except Exception as e:
        print(e)
        return None

    result = {"key": key, "prompt": user_prompt, "response": response}

    with lock:
        with open(output_jsonl_file, "a", encoding="utf-8") as f_out:
            json.dump(result, f_out, ensure_ascii=False)
            f_out.write("\n")

    return result


def answer_prompts():
    total_tasks = len(queries)
    print(f"Total tasks: {total_tasks}")

    lock = Lock()
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_prompt, q, lock) for q in queries]

        for fut in tqdm(as_completed(futures), total=total_tasks, desc="Processing Prompts"):
            _ = fut.result()  

    print(f"Results saved to {output_jsonl_file}")


if __name__ == "__main__":
    answer_prompts()
    print("Response generation completed.")