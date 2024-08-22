import os
import regex
import json
from typing import List, Union
from .base_template import BaseRMTemplate, parse_conversation

class IndividualRMFeedback(BaseRMTemplate):
    template_path = os.path.join(os.path.dirname(__file__), "individual_feedback.template")
    def __init__(self, template_path=None):
        super().__init__(template_path)

    def apply_template(self, feedback: Union[str, dict]):
        return super().apply_template(feedback)
        

class IndividualRMShot(BaseRMTemplate):
    template_path = os.path.join(os.path.dirname(__file__), "individual_shot.template")
    def __init__(self, template_path=None):
        super().__init__(template_path)
        
    def apply_template(self, index:int, conversation: List[dict], feedback: Union[str, dict]):
        history, user_query, model_output = parse_conversation(conversation)
        template_keys = {
            "index": index,
            "history": history or "N/A",
            "user_query": user_query,
            "model_output": model_output,
            "feedback": feedback
        }
        return super().apply_template(**template_keys)

class IndividualRMTemplate(BaseRMTemplate):
    template_path = os.path.join(os.path.dirname(__file__), "individual_overall.template")
    def __init__(self, template_path=None, shot_template_path=None, feedback_template_path=None):
        self.shot_template = IndividualRMShot(shot_template_path)
        self.feedback_template = IndividualRMFeedback(feedback_template_path)
        super().__init__(template_path)

    def apply_template(self, conversation: List[dict], shots: List[dict]):
        shots_strs = [self.shot_template.apply_template(i+1, shot["conversation"], shot["feedback"]) for i, shot in enumerate(shots)]
        shots_str = "\n".join(shots_strs).strip('\n ')
        
        history, user_query, model_output = parse_conversation(conversation)
        template_keys = {
            "history": history or "N/A",
            "user_query": user_query,
            "model_output": model_output,
            "shots": shots_str
        }

        return super().apply_template(**template_keys)
    

if __name__ == "__main__":
    conversation = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm good, how can I help you?"},
        {"role": "user", "content": "I want to book a flight"},
        {"role": "assistant", "content": "Sure, where do you want to go?"}
    ]
    shot = {
        "conversation": [
            {"role": "user", "content": "I want to book a flight"},
            {"role": "assistant", "content": "Sure, where do you want to go?"}
        ],
        "feedback": "The assistant is helpful and responsive"
    }
    shots = [shot]
    template = IndividualRMTemplate()
    print(template.apply_template(conversation, shots*2))