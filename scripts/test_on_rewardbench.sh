num_shots=3
for rm_type in "individual" "pairwise";
do 
    # 128k base models
    python test_on_rewardbench.py --model_name="meta-llama/Meta-Llama-3.1-8B" --rm_type="${rm_type}" --num_workers=8 --num_gpu_per_worker=1 ---num_shots=${num_shots} --engine="vllm"
    # python test_on_rewardbench.py --model_name="meta-llama/Meta-Llama-3.1-70B" --rm_type="${rm_type}" --num_workers=8 --num_gpu_per_worker=1 ---num_shots=${num_shots} --engine="vllm"

    python test_on_rewardbench.py --model_name="Qwen/Qwen2-0.5B" --rm_type="${rm_type}" --num_workers=8 --num_gpu_per_worker=1 ---num_shots=${num_shots} --engine="vllm"
    python test_on_rewardbench.py --model_name="Qwen/Qwen2-1.5B" --rm_type="${rm_type}" --num_workers=8 --num_gpu_per_worker=1 ---num_shots=${num_shots} --engine="vllm"
    python test_on_rewardbench.py --model_name="Qwen/Qwen2-7B" --rm_type="${rm_type}" --num_workers=8 --num_gpu_per_worker=1 ---num_shots=${num_shots} --engine="vllm"
    # python test_on_rewardbench.py --model_name="Qwen/Qwen2-57B-A14B" --rm_type="${rm_type}" --num_workers=2 --num_gpu_per_worker=4 ---num_shots=${num_shots} --engine="vllm"
    python test_on_rewardbench.py --model_name="Qwen/Qwen2-72B" --rm_type="${rm_type}" --num_workers=1 --num_gpu_per_worker=8 ---num_shots=${num_shots} --engine="vllm"


    ## 8K base models

    python test_on_rewardbench.py --model_name="google/gemma-2-2b" --rm_type="${rm_type}" --num_workers=8 --num_gpu_per_worker=1 ---num_shots=${num_shots} --engine="vllm"
    # python test_on_rewardbench.py --model_name="google/gemma-2-9b" --rm_type="${rm_type}" --num_workers=8 --num_gpu_per_worker=1 ---num_shots=${num_shots} --engine="vllm"
    # python test_on_rewardbench.py --model_name="google/gemma-2-27b" --rm_type="${rm_type}" --num_workers=2 --num_gpu_per_worker=4 ---num_shots=${num_shots} --engine="vllm"

    python test_on_rewardbench.py --model_name="meta-llama/Meta-Llama-3-8B" --rm_type="${rm_type}" --num_workers=8 --num_gpu_per_worker=1 ---num_shots=${num_shots} --engine="vllm"

    # other 
    # python test_on_rewardbench.py --model_name="gpt-4o-mini" --rm_type="${rm_type}" --num_workers=8 --num_gpu_per_worker=1 ---num_shots=${num_shots} --engine="openai"
done

for rm_type in "individual" "pairwise";
do 
    # 128k base models
    python test_on_rewardbench.py --model_name="meta-llama/Meta-Llama-3.1-70B" --rm_type="${rm_type}" --num_workers=8 --num_gpu_per_worker=1 ---num_shots=${num_shots} --engine="vllm"
    # python test_on_rewardbench.py --model_name="meta-llama/Meta-Llama-3-70B" --rm_type="${rm_type}" --num_workers=8 --num_gpu_per_worker=1 ---num_shots=${num_shots} --engine="vllm"
    # python test_on_rewardbench.py --model_name="Qwen/Qwen2-72B" --rm_type="${rm_type}" --num_workers=1 --num_gpu_per_worker=8 ---num_shots=${num_shots} --engine="vllm"
    python test_on_rewardbench.py --model_name="Qwen/Qwen2-57B-A14B" --rm_type="${rm_type}" --num_workers=8 --num_gpu_per_worker=1 ---num_shots=${num_shots} --engine="vllm"
done


