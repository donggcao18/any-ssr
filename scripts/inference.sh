#!bin/bash
BASE_PATH="/U_PZL2023ZZ0005/rhe/Any-SSR/output_models"
port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:2 --master_port $port inference/infer_anyssr_total.py \
   --router_weight_path /U_PZL2023ZZ0005/rhe/Any-SSR/output_models/router_weights \
   --data_path /U_PZL2023ZZ0005/rhe/dataset/TRACE-Benchmark/LLM-CL-Benchmark_5000/ \
   --inference_tasks NumGLUE-cm,NumGLUE-ds,FOMC,20Minuten,C-STANCE,Py150,MeetingBank,ScienceQA \
   --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
   --inference_model_path "$BASE_PATH/NumGLUE-cm/0","$BASE_PATH/NumGLUE-ds/0","$BASE_PATH/FOMC/0","$BASE_PATH/20Minuten/0","$BASE_PATH/C-STANCE/0","$BASE_PATH/Py150/0","$BASE_PATH/MeetingBank/0","$BASE_PATH/ScienceQA/0" \
   --max_prompt_len 1024 \
   --max_ans_len 512 \
   --seed 1234 \
   --deepspeed \
   --inference_output_path /U_PZL2023ZZ0005/rhe/Any-SSR/inference_result

