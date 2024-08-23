# IC-RM

## Installation
```bash
pip install -e .
pip install git+https://github.com/jdf-prog/LLM-Engines.git
pip install vllm==0.5.3.post1 # for now
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ # required for sglang
pip install flash-attn --no-build-isolation
```

## Usage
- simple cli example:
```bash
icrm test ./examples/test_individual.json --rm_type individual --model_name meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --engine together
icrm test ./examples/test_individual.json --rm_type individual --model_name meta-llama/Meta-Llama-3-8B-Instruct --engine sglang
icrm test ./examples/test_pairwise.json   --rm_type pairwise   --model_name meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --engine together
icrm test ./examples/test_pairwise.json   --rm_type pairwise   --model_name meta-llama/Meta-Llama-3-8B-Instruct --engine sglang 
```

### use in python:

- **individual scoring example**


```python
inputs = {
    "conversation": [
        {"role": "user", "content": "I want to book a flight"},
        {"role": "assistant", "content": "Sure, where do you want to go?"}
    ]
}
from icrm import ICRM
model = ICRM(model_name='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', rm_type='individual', engine='together')
model = ICRM(model_name='meta-llama/Meta-Llama-3-8B-Instruct', rm_type='individual', engine='sglang')
generation_kwargs = {
    "max_tokens": 3072,
    "temperature": 0.7,
    "top_p": 0.9,
}
feedback = model(**inputs, **generation_kwargs)
print(feedback)
"""
{
    "intent": "The user is looking to book a flight and is seeking assistance with the process. This typically involves gathering information about the user's travel plans, such as destination, travel dates, and possibly preferred airlines or flight classes.",
    "checklist": "1. Engagement with the user's intent; 2. Request for necessary information; 3. Clarity of the response; 4. Guidance on the next steps; 5. Overall helpfulness.",
    "strengths": "The response acknowledges the user's intent to book a flight and asks for a crucial piece of information (destination) to proceed. This shows engagement with the user's request and a willingness to assist.",
    "weaknesses": "The response is very brief and does not provide any additional guidance on the next steps in the booking process. It also does not offer any options for the user to specify their travel dates, preferred airlines, or other relevant details that are typically necessary for booking a flight.",
    "score": 4
}
"""
```

- **pairwise scoring example**

```python
inputs = {
    "conversation_A": [
        {"role": "user", "content": "I want to book a flight"},
        {"role": "assistant", "content": "Sure, where do you want to go?"}
    ],
    "conversation_B": [
        {"role": "user", "content": "I want to book a flight"},
        {"role": "assistant", "content": "To help you book a flight, I'll need some more information:\n\n1. What are your departure and arrival cities?\n2. What are your preferred travel dates?\n3. Do you need a one-way or round-trip ticket?\n4. How many passengers will be traveling?\n5. Do you have any preferences for airlines, flight times, or class of service?\n\nOnce you provide these details, I can offer more specific guidance on booking your flight. Is there any other information you'd like to share about your travel plans?"}
    ]
}

from icrm import ICRM
# model = ICRM(model_name='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', rm_type='pairwise', engine='together')
model = ICRM(model_name='meta-llama/Meta-Llama-3-8B-Instruct', rm_type='pairwise', engine='sglang')
generation_kwargs = {
    "max_tokens": 3072,
    "temperature": 0.7,
    "top_p": 0.9,
}
feedback = model(**inputs, **generation_kwargs)
print(feedback)
"""
{
    "analysis of A": "Response A is brief and to the point, asking for the destination. However, it lacks the necessary follow-up questions to gather more information about the user's preferences and needs.",
    "analysis of B": "Response B is more comprehensive, asking for specific details about the user's travel plans, including departure and arrival cities, travel dates, and preferences for airlines and flight times. This shows a better understanding of the user's needs and provides a more personalized experience.",
    "reason of A=B": "Both responses acknowledge the user's request to book a flight and ask for some information.",
    "reason of A>B": "Response A is more concise and may be preferred by users who want a quick and simple interaction.",
    "reason of B>A": "Response B is more detailed and shows a better understanding of the user's needs, making it more likely to provide a successful booking experience.",
    "choice": "B+"
}
"""
```

### Init parameters of ICRM
- `model_name`: The name of the model to use (e.g., 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo').
- `shots_pool_path`: Optional path to a pool of shots for few-shot learning. (default files are at [icrm/shots_pool](icrm/shots_pool))
- `rm_type`: The type of scoring to use, either 'individual' or 'pairwise'. (default is 'individual')
- `engine`: The engine to use for inference (default is 'vllm'). one of ['together', 'sglang', 'vllm', 'openai', 'mistral', 'claude']
- `num_workers`: The number of workers to use for inference (default is 1).
- `num_gpu_per_worker`: The number of GPUs to allocate per worker (default is 1).
- `use_cache`: Whether to use cache for inference (default is False).
- `completion`: Whether to use completion mode (default is True). We recommend to use base model for completion for better few-shot performance.
- `num_shots`: The number of shots sampled from the pool for few-shot learning (default is 3).
- `template_path`: Optional path to a template for formatting inputs, default is [icrm/templates/individual_overall.template](icrm/templates/individual_overall.template).
- `shot_template_path`: Optional path to a template for formatting shots, default is [icrm/templates/individual_shot.template](icrm/templates/individual_shot.template).
- `feedback_template_path`: Optional path to a template for formatting feedback, default is [icrm/templates/inidividual_feedback.template](icrm/templates/inidividual_feedback.template).
- `verbose`: Whether to print verbose output (default is False).

### Inference inputs
- \*\*inputs: A dictionary. Refer to the examples for the structure of the input data.
- \*\*generation_kwargs: A dictionary of generation parameters for the model. Refer to [LLM-Engines](https://github.com/jdf-prog/LLM-Engines?tab=readme-ov-file#generation-parameters) for more details on the available parameters.
- shots (List[str]): Optional list of shots for few-shot learning. If not provided, the model will sample from the shots pool. Please refer to the `icrm/shots_pool` for the structure of the shots. 
    You can also use `icrm.sample_shots` to sample shots from the pool to inspect the structure.
- parse_json (bool): Whether to parse the output as JSON. Default is False. 

### Output
- The output will be a pure string. By default it's a json string containing the feedback score and other relevant information. You can parse it as needed.

### How to maximize the inference speed?
ICRM is based on [LLM-Engines](https://github.com/jdf-prog/LLM-Engines) where you can totally view each inference as a async api call. To maximize the inference speed, we recommand you to use multiple processes. 

For example you can use `datasets.map` to parallelize the inference across multiple examples. 

```python
import datasets
inputs = [...]
dataset = datasets.Dataset.from_dict({"inputs": inputs})
model = ICRM(model_name='meta-llama/Meta-Llama-3-8B-Instruct', rm_type='individual', engine='sglang', num_workers=4, num_gpu_per_worker=1)
def inference_fn(inputs):
    return model(**inputs, max_tokens=3072, temperature=0.7, top_p=0.9)

results = dataset.map(inference_fn, batched=True, num_proc=4 * 8) # 8 means each model worker can be assigned to 8 requests in parallel
```


## Evaluation
see [scripts/test_on_rewardbench.sh](scripts/test_on_rewardbench.sh) for details.

### IC-RM Results on Reward-Bench
todo




