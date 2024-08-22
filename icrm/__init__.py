import os
import json
from .templates import IndividualRMTemplate, PairwiseRMTemplate

class ICRM:
    default_icrm_pool_path = os.path.join(os.path.dirname(__file__), "icrm_pool.json")
    def __init__(
        self, 
        model_name: str, 
        icrm_examples = None,
        engine:str="vllm",
        num_workers:int=1,
        num_gpu_per_worker:int=1,
        use_cache:bool=True,
        completion:bool=True
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
        from llm_engines import get_call_worker_func, workers, cleanup_process
        self.model_name = model_name
        self.engine = engine
        self.num_workers = num_workers
        self.num_gpu_per_worker = num_gpu_per_worker
        self.use_cache = use_cache
        if icrm_examples is not None:
            self.icrm_examples = icrm_examples
        else:
            self.icrm_examples = DEFAULT_ICRM_EXAMPLES
        
        _previous_workers = workers.copy()
        self.call_worker = get_call_worker_func(
            model_name,
            engine=engine, 
            num_workers=num_workers, 
            num_gpu_per_worker=num_gpu_per_worker, 
            use_cache=use_cache,
            completion=completion
        )
        self.workers = [worker for worker in workers if worker not in _previous_workers]

    # cleanup the workers after the class is deleted
    def __del__(self):
        for worker in self.workers:
            cleanup_process(worker)

    def __call__(self, messages):

        return self.call_worker(messages)

