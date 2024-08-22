import json
import regex
from typing import List, Union

def apply_template(template, **kwargs):
    # {$key} -> value
    for key, value in kwargs.items():
        template = template.replace("{$" + key + "}", value)
    return template

def parse_conversation(conversation: List[dict], user_role="user", assistant_role="assistant", system_role="system"):
    assert len(conversation) >= 2, "The conversation should have at least 2 turns"
    if conversation[0]["role"] == user_role:
        for i in range(len(conversation)):
            if i % 2 == 0:
                assert conversation[i]["role"].lower() == user_role, f"Expecting `user` role, but got {conversation[i]['role'].lower()} at turn {i}"
            else:
                assert conversation[i]["role"].lower() == assistant_role, f"Expecting `assistant` role, but got {conversation[i]['role'].lower()} at turn {i}"
    elif conversation[0]["role"] == system_role:
        for i in range(1, len(conversation)):
            if i % 2 == 0:
                assert conversation[i]["role"] == assistant_role, f"Expecting `assistant` role, but got {conversation[i]['role'].lower()} at turn {i}"
            else:
                assert conversation[i]["role"] == user_role, f"Expecting `user` role, but got {conversation[i]['role'].lower()} at turn {i}"
    else:
        raise ValueError("The first role should be either user or system")
    
    history = conversation[:-2]
    history = "\n\n".join([f"{turn['role'].upper()}: {turn['content']}" for turn in history])
    user_query = conversation[-2]["content"]
    model_output = conversation[-1]["content"]
    return history, user_query, model_output

class BaseRMTemplate:
    template_path = None
    def __init__(self, template_path=None):
        if template_path is not None:
            self.template_path = template_path
        with open(self.template_path) as f:
            self.template = f.read()

        if self.template:
            # {$key} -> value
            required_keys = [m.group(1) for m in regex.finditer(r"\{\$(\w+)\}", self.template)]
            self.required_keys = required_keys
        else:
            self.required_keys = []
    
    def apply_template(self, **kwargs):
        if not self.required_keys:
            return json.dumps(kwargs)
        else:
            missing_keys = [key for key in self.required_keys if key not in kwargs]
            if missing_keys:
                raise ValueError(f"Missing required keys: {missing_keys}; provided keys: {kwargs.keys()}")
            return apply_template(self.template, **{x:str(s) for x,s in kwargs.items()})