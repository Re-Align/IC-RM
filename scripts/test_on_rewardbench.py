import fire
import datasets
import json
import random
from icrm import ICRM
from pathlib import Path

def load_rewardbench(dataset="allenai/reward-bench", split='filtered'):
    dataset = datasets.load_dataset(dataset, split=split)
    def format_fn(item):
        return {
            "chosen": {
                "conversation": [
                    {
                        "role": "user",
                        "content": item['prompt']
                    },
                    {
                        "role": "assistant",
                        "content": item['chosen']
                    }
                ]
            },
            "rejected": {
                "conversation": [
                    {
                        "role": "user",
                        "content": item['prompt']
                    },
                    {
                        "role": "assistant",
                        "content": item['rejected']
                    }
                ]
            },
            "subset": item['subset'],
            "id": item['id']
        }
    dataset = dataset.map(format_fn, remove_columns=dataset.column_names)
    return dataset

def main(
    model_name='meta-llama/Meta-Llama-3-8B-Instruct',
    shots_pool_path=None,
    rm_type='individual',
    engine='vllm',
    dtype:str='auto',
    quantization=None,
    num_workers=1,
    num_gpu_per_worker=1,
    use_cache=True,
    overwrite_cache=False,
    completion=True,
    num_shots=3,
    results_dir=None,
    overwrite_results=False,
    seed=42,
    shots_sampling_mode="fixed",
    verbose=False
):
    if seed is not None:
        random.seed(seed)
        
    if results_dir is None:
        results_dir = f"results"
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    results_file = results_dir / rm_type / f"{model_name.replace('/', '_')}" / f"{num_shots}_shots.json"
    
    dataset = load_rewardbench()
    
    if results_file.exists() and not overwrite_results:
        existing_results = datasets.Dataset.from_json(str(results_file))
        if set(existing_results.unique("id")) == set(dataset.unique("id")):
            print(f"Results file already exists: {results_file}; if you want to overwrite it, set overwrite_results=True")
            dataset = existing_results
            do_rm = False
        else:
            print(f"Results file already exists: {results_file}; but the ids don't match, overwriting")
            do_rm = True
    else:
        do_rm = True
        
    if do_rm:
        model = ICRM(
            model_name=model_name,
            shots_pool_path=shots_pool_path,
            rm_type=rm_type,
            engine=engine,
            dtype=dtype,
            quantization=quantization,
            num_workers=num_workers,
            num_gpu_per_worker=num_gpu_per_worker,
            use_cache=use_cache,
            overwrite_cache=overwrite_cache,
            completion=completion,
            num_shots=num_shots,
            shot_sampling_mode=shots_sampling_mode,
            verbose=verbose
        )
        all_shots = [model.sample_shots(num_shots=num_shots) for _ in range(len(dataset))]
        dataset = dataset.add_column("shots", all_shots)

        generation_kwargs = {
            "max_tokens": 3072, # or 2048
            "top_p": 1.0,
            "temperature": 0.0,
            "parse_json": True,
            "timeout": 300,
        }
        

        def map_fn_individual(item):
            chosen_inputs = {
                "conversation": item['chosen']['conversation'],
                "shots": item['shots']
            }
            rejected_inputs = {
                "conversation": item['rejected']['conversation'],
                "shots": item['shots']
            }
            chosen_result = model(**chosen_inputs, **generation_kwargs)
            rejected_result = model(**rejected_inputs, **generation_kwargs)
            is_correct = None
            try:
                chosen_score = int(chosen_result['score'])
                rejected_score = int(rejected_result['score'])
                is_correct = chosen_score > rejected_score
            except Exception as e:
                print(e)
                print(chosen_result)
                print(rejected_result)
            item['is_correct'] = is_correct
            item['feedback'] = json.dumps({
                "chosen": chosen_result,
                "rejected": rejected_result
            })
            return item
        
        def map_fn_pairwise(item):
            inputs = {
                "shots": item['shots'],
                "conversation_A": item['chosen']['conversation'],
                "conversation_B": item['rejected']['conversation']
            }
            result = model(**inputs, **generation_kwargs)
            is_correct = None
            try:
                choice = result['choice']
                if choice in ["A+", "A++"]:
                    is_correct = True
                else:
                    is_correct = False
            except Exception as e:
                print(e)
            item['is_correct'] = is_correct
            item['feedback'] = json.dumps(result)
            return item
        
        if rm_type == "individual":
            dataset = dataset.map(map_fn_individual, num_proc=num_workers * 8)
        elif rm_type == "pairwise":
            dataset = dataset.map(map_fn_pairwise, num_proc=num_workers * 8)
        else:
            raise ValueError(f"Unknown rm_type: {rm_type}")
    
    # disable progress bar
    datasets.disable_progress_bar()
    # calculate accuracy
    accuracy = dataset.filter(lambda x: x['is_correct']).num_rows / dataset.num_rows
    print(f"Accuracy: {accuracy}")
    
    # per subset accuracy
    acc_results = {}
    all_subsets = dataset.unique("subset")
    for subset in all_subsets:
        subset_accuracy = dataset.filter(lambda x: x['subset'] == subset).filter(lambda x: x['is_correct']).num_rows / dataset.filter(lambda x: x['subset'] == subset).num_rows
        print(f"Subset: {subset}; Accuracy: {subset_accuracy}")
        acc_results[subset] = subset_accuracy
    acc_results['overall'] = sum(acc_results.values()) / len(acc_results)
    print(f"Overall Accuracy: {acc_results['overall']}")

    results_file.parent.mkdir(exist_ok=True, parents=True)
    with open(results_file, 'w') as f:
        json.dump([x for x in dataset], f, indent=4)
    print(f"Results saved to: {results_file}")
    
    meta_info = {}
    meta_info['model_name'] = model_name
    meta_info['num_shots'] = num_shots
    meta_info['rm_type'] = rm_type
    meta_info['engine'] = engine
    meta_info['completion'] = completion
    acc_results['meta_info'] = meta_info
    
    acc_results_file = results_file.parent / "accuracy.json"
    if acc_results_file.exists():
        with open(acc_results_file, 'r') as f:
            existing_acc_results = json.load(f)
    else:
        existing_acc_results = {}
    existing_acc_results[num_shots] = acc_results
    with open(acc_results_file, 'w') as f:
        json.dump(existing_acc_results, f, indent=4)
    
if __name__ == "__main__":
    fire.Fire(main)

"""
rm_type="individual"
num_shots=3
# 128k base models
python test_on_rewardbench.py --model_name="meta-llama/Meta-Llama-3.1-8B" --rm_type="${rm_type}" --num_workers=8 --num_gpu_per_worker=1 ---num_shots=${num_shots} --engine="vllm"
python test_on_rewardbench.py --model_name="meta-llama/Meta-Llama-3.1-70B-Reference" --rm_type="${rm_type}" --num_workers=8 --num_gpu_per_worker=1 ---num_shots=${num_shots} --engine="together"

python test_on_rewardbench.py --model_name="Qwen/Qwen2-0.5B" --rm_type="${rm_type}" --num_workers=4 --num_gpu_per_worker=2 ---num_shots=${num_shots} --engine="vllm"
python test_on_rewardbench.py --model_name="Qwen/Qwen2-1.5B" --rm_type="${rm_type}" --num_workers=4 --num_gpu_per_worker=2 ---num_shots=${num_shots} --engine="vllm"
python test_on_rewardbench.py --model_name="Qwen/Qwen2-7B" --rm_type="${rm_type}" --num_workers=4 --num_gpu_per_worker=2 ---num_shots=${num_shots} --engine="vllm"
python test_on_rewardbench.py --model_name="Qwen/Qwen2-57B-A14B" --rm_type="${rm_type}" --num_workers=2 --num_gpu_per_worker=4 ---num_shots=${num_shots} --engine="vllm"
python test_on_rewardbench.py --model_name="Qwen/Qwen2-72B" --rm_type="${rm_type}" --num_workers=1 --num_gpu_per_worker=8 ---num_shots=${num_shots} --engine="vllm"


## 8K base models

python test_on_rewardbench.py --model_name="google/gemma-2-2b" --rm_type="${rm_type}" --num_workers=4 --num_gpu_per_worker=2 ---num_shots=${num_shots} --engine="vllm"
python test_on_rewardbench.py --model_name="google/gemma-2-9b" --rm_type="${rm_type}" --num_workers=4 --num_gpu_per_worker=2 ---num_shots=${num_shots} --engine="vllm"
python test_on_rewardbench.py --model_name="google/gemma-2-27b" --rm_type="${rm_type}" --num_workers=4 --num_gpu_per_worker=2 ---num_shots=${num_shots} --engine="vllm"

python test_on_rewardbench.py --model_name="meta-llama/Meta-Llama-3-8B" --rm_type="${rm_type}" --num_workers=8 --num_gpu_per_worker=1 ---num_shots=${num_shots} --engine="vllm"

# other 
python test_on_rewardbench.py --model_name="gpt-4o-mini" --rm_type="${rm_type}" --num_workers=8 --num_gpu_per_worker=1 ---num_shots=${num_shots} --engine="openai"




# instruct models (should not use instructed models for ICRM, they don't work well)
python test_on_rewardbench.py --model_name="meta-llama/Meta-Llama-3-8B-Instruct" --rm_type="${rm_type}" --num_workers=8 --num_gpu_per_worker=1 --completion=False ---num_shots=${num_shots} --engine="vllm"
python test_on_rewardbench.py --model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" --rm_type="${rm_type}" --num_workers=8 --num_gpu_per_worker=1 --completion=False ---num_shots=${num_shots} --engine="together"

"""