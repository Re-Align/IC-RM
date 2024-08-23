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


"""
def __init__(
        self, 
        model_name: str, 
        shots_pool_path: Union[str, None] = None,
        rm_type: str = "individual",
        engine:str="vllm",
        num_workers:int=1,
        num_gpu_per_worker:int=1,
        use_cache:bool=False,
        completion:bool=True,
        num_shots:int=3,
        template_path:str=None,
        shot_template_path:str=None,
        feedback_template_path:str=None,
        verbose: bool = False,
    ):
    """
def main(
    model_name='meta-llama/Meta-Llama-3-8B-Instruct',
    shots_pool_path=None,
    rm_type='individual',
    engine='vllm',
    num_workers=1,
    num_gpu_per_worker=1,
    use_cache=True,
    overwrite_cache=False,
    completion=True,
    num_shots=3,
    results_dir=None,
    overwrite_results=False,
    seed=42,
    shots_sampling_mode="fixed"
):
    
    if results_dir is None:
        results_dir = f"results"
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    results_file = results_dir / rm_type / f"{model_name.replace('/', '_')}" / f"{num_shots}_shots.json"
    if results_file.exists() and not overwrite_results:
        print(f"Results file already exists: {results_file}; if you want to overwrite it, set overwrite_results=True")
        return
    model = ICRM(
        model_name=model_name,
        shots_pool_path=shots_pool_path,
        rm_type=rm_type,
        engine=engine,
        num_workers=num_workers,
        num_gpu_per_worker=num_gpu_per_worker,
        use_cache=use_cache,
        overwrite_cache=overwrite_cache,
        completion=completion,
        num_shots=num_shots,
        shot_sampling_mode=shots_sampling_mode,
    )
    dataset = load_rewardbench()

    generation_kwargs = {
        "max_tokens": None, # or 2048
        "top_p": 1.0,
        "temperature": 0.0,
        "parse_json": True,
        "timeout": 300,
    }
    if seed is not None:
        random.seed(seed)
    all_shots = []
    def map_assign_shots(item):
        item['shots'] = model.sample_shots(num_shots=num_shots)
        return item
    dataset = dataset.map(map_assign_shots)


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
        is_correct = False
        try:
            chosen_score = int(chosen_result['score'])
            rejected_score = int(rejected_result['score'])
            is_correct = chosen_score > rejected_score
        except Exception as e:
            print(e)
            is_correct = False
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
        result = model.icrm(**inputs, **generation_kwargs)
        is_correct = False
        try:
            choice = result['choice']
            if choice in ["A+", "A++"]:
                is_correct = True
            else:
                is_correct = False
        except Exception as e:
            print(e)
            is_correct = False
        item['is_correct'] = is_correct
        item['feedback'] = json.dumps(result)
        return item
    
    if rm_type == "individual":
        dataset = dataset.map(map_fn_individual, num_proc=num_workers * 8)
    elif rm_type == "pairwise":
        dataset = dataset.map(map_fn_pairwise, num_proc=num_workers * 8)
    else:
        raise ValueError(f"Unknown rm_type: {rm_type}")
    
    # calculate accuracy
    accuracy = dataset.filter(lambda x: x['is_correct']).num_rows / dataset.num_rows
    print(f"Accuracy: {accuracy}")

    dataset.to_json(str(results_file))
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    fire.Fire(main)

"""
python test_on_rewardbench.py --model_name="meta-llama/Meta-Llama-3-8B-Instruct" --rm_type="individual" --num_workers=8 --num_gpu_per_worker=1 --use_cache True --completion=True --num_shots=3
"""