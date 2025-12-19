#!/bin/bash

input_data=assets/ifeval_pp_verified.jsonl
output_dir=results/ifeval_pp
model_name=gpt-4.1-mini

(
python generate_response.py \
    --model=$model_name \
    --client_type=oai \
    --input_data=$input_data \
    --output_dir=$output_dir/responses \
    --num_workers=512 \
    --runname=$model_name

python evaluation_main.py \
    --input_data=$input_data \
    --input_response_data=$output_dir/responses/$model_name.jsonl \
    --output_dir=$output_dir/evaluation \
    --model_name=$model_name
) &

wait