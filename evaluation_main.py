# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary of evaluating instruction following. See README.md."""

import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

from src import evaluation_lib as evaluation_lib

from datasets import load_dataset
ds = load_dataset("RebeccaYU920/ifeval-pp")


_INPUT_DATA = flags.DEFINE_string(
    "input_data", "assets/ifeval_pp.jsonl", "path to input data"
)

_INPUT_RESPONSE_DATA = flags.DEFINE_string(
    "input_response_data", "results/ifeval_pp/responses/gpt-4.1-mini.jsonl", "path to input response data"
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "results/ifeval_pp/evaluation",
    "Output directory for inference and eval results."
)

_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "gpt-4.1-mini",
    "Name of the model used for evaluation.",
)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  inputs = evaluation_lib.read_prompt_list(_INPUT_DATA.value)
  prompt_to_response = evaluation_lib.read_prompt_to_response_dict(
      _INPUT_RESPONSE_DATA.value)

  # get instruction following results
  for func, output_name in [
      (evaluation_lib.test_instruction_following_strict, "eval_results_strict"),
  ]:
    logging.info("Generating %s...", output_name)
    outputs = []
    for inp in inputs:
      outputs.append(func(inp, prompt_to_response))
    follow_all_instructions = [o.follow_all_instructions for o in outputs]
    
    accuracy = sum(follow_all_instructions) / len(outputs)
    logging.info("Accuracy: %f", accuracy)
    try:
        evaluation_lib.calculate_reliable_at_k(outputs)
    except Exception as e:
        pass
    evaluation_lib.print_report(outputs) # !!!
    
    # extract responses_sys0.jsonl   
    file_name = _INPUT_RESPONSE_DATA.value.split("/")[-1]
    
    output_dir = os.path.join(
        _OUTPUT_DIR.value, "stat_" + _MODEL_NAME.value
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    core_name = file_name.replace("input_response_data_", "").replace(".jsonl", "")
    
    output_file_name = os.path.join(
        output_dir, 'raw', f"{core_name}_{output_name}_detail.jsonl"
    )
    if not os.path.exists(os.path.dirname(output_file_name)):
        os.makedirs(os.path.dirname(output_file_name))
    
    evaluation_lib.write_outputs(output_file_name, outputs)
    print(f"Results saved to {output_file_name}")

    log_path = os.path.join(
        output_dir, 'log', f"{core_name}_{output_name}_log.json"
    )
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
        
    evaluation_lib.write_log(outputs, log_path)
    print(f"Log saved to {log_path}")
    
    print("=" * 64)


if __name__ == "__main__":
  app.run(main)