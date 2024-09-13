import fire
import datasets
import json
import random
from icrm import ICRM
from pathlib import Path
from collections import defaultdict

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
    verbose=False,
    max_tokens:int=None,
):
    if seed is not None:
        random.seed(seed)
        
    if results_dir is None:
        results_dir = f"results"
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    results_file = results_dir / rm_type / f"{model_name.replace('/', '_')}" / f"{num_shots}_shots.json"
    
    dataset = load_rewardbench()
    # dataset = dataset.select(range(100)) # for testing

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
            "max_tokens": int(max_tokens) if max_tokens is not None else None,
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
        
        def map_fn_pairwise(item, index):
            
            if index % 2 == 0:
                inputs = {
                    "shots": item['shots'],
                    "conversation_B": item['chosen']['conversation'],
                    "conversation_A": item['rejected']['conversation']
                }
                winner = "B"
            else:
                inputs = {
                    "shots": item['shots'],
                    "conversation_A": item['chosen']['conversation'],
                    "conversation_B": item['rejected']['conversation']
                }
                winner = "A"
            if verbose:
                print("-----" * 10)
                print(f"Index: {index}")
                print(f"Winner: {winner}")
            result = model(**inputs, **generation_kwargs)
            if verbose:
                print(f"map Result: {result}")
            is_correct = None
            try:
                choice = result['choice']
                if winner == "A":
                    winner_choices = ["A+", "A++"]
                else:
                    winner_choices = ["B+", "B++"]
                if choice in winner_choices:
                    is_correct = True
                else:
                    is_correct = False
            except Exception as e:
                pass
            if verbose:
                print(f"Index: {index}; is_correct: {is_correct}")
            item['is_correct'] = is_correct
            item['winner'] = winner
            item['feedback'] = json.dumps(result)
            return item
        
        if rm_type == "individual":
            dataset = dataset.map(map_fn_individual, num_proc=num_workers*8)
        elif rm_type == "pairwise":
            dataset = dataset.map(map_fn_pairwise, num_proc=num_workers*8, with_indices=True)
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
    
    

    if rm_type == 'pairwise':
        # count the distribution of the choices
        choice_distribution = {"winner is A": {} , "winner is B": {}}
        for item in dataset:
            try:
                feedback = json.loads(item['feedback'])
                winner = item['winner']
                choice = feedback['choice']
            except Exception as e:
                choice = "error"
            if winner == "A":
                choice_distribution["winner is A"][choice] = choice_distribution["winner is A"].get(choice, 0) + 1
            else:
                choice_distribution["winner is B"][choice] = choice_distribution["winner is B"].get(choice, 0) + 1
        # sort chioce distribution dict
        for k, v in choice_distribution.items():
            choice_distribution[k] = dict(sorted(v.items(), key=lambda x: x[0]))
        acc_results['choice_distribution'] = choice_distribution
    elif rm_type == 'individual':
        score_distribution = defaultdict(lambda: defaultdict(int))
        choice_distribution = {}
        for item in dataset:
            try:
                feedback = json.loads(item['feedback'])
                chosen_score = int(feedback['chosen']['score'])
            except Exception as e:
                chosen_score = None
            try:
                rejected_score = int(feedback['rejected']['score'])
            except Exception as e:
                rejected_score = None
            if chosen_score is None or rejected_score is None:
                choice = "error"
            else:
                if chosen_score > rejected_score:
                    choice = "chosen"
                elif chosen_score < rejected_score:
                    choice = "rejected"
                else:
                    choice = "tie"
            score_distribution[chosen_score][rejected_score] += 1
            
        # sort the score distribution dict
        for k, v in score_distribution.items():
            score_distribution[k] = dict(sorted(v.items(), key=lambda x: x[0] if x[0] is not None else -1))
        score_distribution = dict(sorted(score_distribution.items(), key=lambda x: x[0] if x[0] is not None else -1))
        # sort chioce distribution dict
        choice_distribution = dict(sorted(choice_distribution.items(), key=lambda x: x[0]))
        acc_results['score_distribution'] = score_distribution
        acc_results['choice_distribution'] = choice_distribution
    else:
        raise ValueError(f"Unknown rm_type: {rm_type}")
    print(f"Choice Distribution: {json.dumps(choice_distribution, indent=4)}")

    acc_results_file = results_file.parent / "accuracy.json"
    if acc_results_file.exists():
        with open(acc_results_file, 'r') as f:
            existing_acc_results = json.load(f)
    else:
        existing_acc_results = {}
    existing_acc_results[f"{num_shots}-shots"] = acc_results
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