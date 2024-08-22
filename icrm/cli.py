import fire
import json
from .icrm import ICRM
from typing import Union

class ICRMCli:
    def __init__(self):
        pass
    
    def test(
        self, 
        example_json_path: str,
        model_name: str, 
        shots_pool_path: Union[str, None] = None,
        rm_type: str = "individual",
        engine:str="vllm",
        num_workers:int=1,
        num_gpu_per_worker:int=1,
        use_cache:bool=False,
        completion:bool=True,
        num_shots:int=5,
        template_path:str=None,
        shot_template_path:str=None,
        feedback_template_path:str=None,
    ):
        icrm = ICRM(
            model_name=model_name,
            shots_pool_path=shots_pool_path,
            rm_type=rm_type,
            engine=engine,
            num_workers=num_workers,
            num_gpu_per_worker=num_gpu_per_worker,
            use_cache=use_cache,
            completion=completion,
            num_shots=num_shots,
            template_path=template_path,
            shot_template_path=shot_template_path,
            feedback_template_path=feedback_template_path,
            verbose=True,
        )
        with open(example_json_path, "r") as f:
            examples = json.load(f)
        # get the conversation from the input
        if rm_type == "individual":
            inputs = {"conversation": examples["conversation"]}
        elif rm_type == "pairwise":
            inputs = {
                "conversation_A": examples["conversation_A"], 
                "conversation_B": examples["conversation_B"]
            }
        else:
            raise ValueError(f"Unknown rm_type: {rm_type}")  
        feedback = icrm(**inputs, max_tokens=3072)
        print(feedback)
    
def main():
    fire.Fire(ICRMCli)

if __name__ == "__main__":
    main()
    
"""
icrm test ./examples/test_individual.json --rm_type individual --model_name meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --engine together
icrm test ./examples/test_individual.json --rm_type individual --model_name meta-llama/Meta-Llama-3-8B-Instruct --engine sglang
icrm test ./examples/test_pairwise.json   --rm_type pairwise   --model_name meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --engine together --rm_type pairwise
icrm test ./examples/test_pairwise.json   --rm_type pairwise   --model_name meta-llama/Meta-Llama-3-8B-Instruct --engine sglang 

"""