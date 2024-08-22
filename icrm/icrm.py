import os
import json
import random
from .templates import IndividualRMTemplate, PairwiseRMTemplate
from typing import List, Dict, Union
from llm_engines import get_call_worker_func, workers, cleanup_process
from .utils import parse_json

class ICRM:
    default_icrm_pool_paths = {
        "individual": os.path.join(os.path.dirname(__file__), "shots_pool", "icrm_pool_individual.json"),
        "pairwise": os.path.join(os.path.dirname(__file__), "shots_pool", "icrm_pool_pairwise.json"),
    }
    default_generation_kwargs = {
        "stop": ["<|end_of_feedback|>"],
        "temperature": 0.0,
        "max_tokens": 3072,
        "top_p": 1.0,
    }
        
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
        Args:
            model_name: the model name to use (recommend to use base model for in-context learning)
            icrm_examples: the ICRM examples to use, format: 
                [
                    {
                        "conversation": [{"role": "...", "content": "..."}, ...], ...],
                        "reward": "..."
                    },
                    ...
                ]
            engine: the engine to use. should be one of ["vllm", "sglang", "openai", "mistral", "claude", "together"], see llm_engines for more details
            num_workers: the number of workers to use
            num_gpu_per_worker: the number of gpus per worker
            use_cache: whether to use
        """
        self.model_name = model_name
        self.engine = engine
        self.num_workers = num_workers
        self.num_gpu_per_worker = num_gpu_per_worker
        self.use_cache = use_cache
        self.completion = completion
        self.rm_type = rm_type
        self.num_shots = num_shots
        self.verbose = verbose
        
        if shots_pool_path is not None:
            with open(shots_pool_path, "r") as f:
                self.icrm_shots_pool = json.load(f)
        else:
            with open(self.default_icrm_pool_paths[rm_type], "r") as f:
                self.icrm_shots_pool = json.load(f)
        
        _previous_workers = workers.copy()
        self.call_worker = get_call_worker_func(
            model_name,
            engine=engine, 
            num_workers=num_workers, 
            num_gpu_per_worker=num_gpu_per_worker, 
            use_cache=use_cache,
            completion=completion,
            overwrite_cache=True,
        )
        self.workers = [worker for worker in workers if worker not in _previous_workers]
        random.seed(42)
        
        self.individual_template = IndividualRMTemplate(template_path, shot_template_path, feedback_template_path)
        self.pairwise_template = PairwiseRMTemplate(template_path, shot_template_path, feedback_template_path)

    # cleanup the workers after the class is deleted
    def __del__(self):
        if hasattr(self, 'workers') and self.workers:
            for worker in self.workers:
                try:
                    cleanup_process(worker)
                except Exception as e:
                    pass   

    def individual_rm(
        self, 
        conversation: List[Dict[str, str]], 
        shots: Union[List[dict], None] = None, 
        parse_json: bool = False,
        **kwargs
    ):
        if shots is None:
            shots = random.sample(self.icrm_shots_pool, self.num_shots)
        prompt = self.individual_template.apply_template(conversation, shots)
        if self.verbose:
            print(f"Prompt for individual RM: \n```\n{prompt}```")
        for key in self.default_generation_kwargs:
            if key not in kwargs:
                kwargs[key] = self.default_generation_kwargs[key]
        result = self.call_worker(prompt, **kwargs)
        if parse_json:
            result = parse_json(result)
        return result
    
    def pairwise_rm(
        self, 
        conversation_A: List[Dict[str, str]], 
        conversation_B: List[Dict[str, str]], 
        shots: Union[List[dict], None] = None,
        parse_json: bool = False,
        **kwargs
    ):
        if shots is None:
            shots = random.sample(self.icrm_shots_pool, self.num_shots)
        prompt = self.pairwise_template.apply_template(conversation_A, conversation_B, shots)
        if self.verbose:
            print(f"Prompt for pairwise RM: \n```\n{prompt}```")
        for key in self.default_generation_kwargs:
            if key not in kwargs:
                kwargs[key] = self.default_generation_kwargs[key]
        result = self.call_worker(prompt, **kwargs)
        if parse_json:
            result = parse_json(result)
        return result
    
    def __call__(self, **kwargs):
        if self.rm_type == "individual":
            return self.individual_rm(**kwargs)
        elif self.rm_type == "pairwise":
            return self.pairwise_rm(**kwargs)
        else:
            raise ValueError(f"Unknown rm_type: {self.rm_type}")
        
    def sample_shots(self, num_shots: int):
        return random.sample(self.icrm_shots_pool, num_shots)

# Example usage:
if __name__ == "__main__":
    icrm = ICRM(model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", engine="together")
    conversation = [{"role": "user", "content": "Hello!"}]
    response = icrm.individual_rm(conversation)
    print(response)
