import os
import regex
import json
from typing import List, Union
from .base_template import BaseRMTemplate, parse_conversation

def parse_conversation_pair(conversation_A: List[dict], conversation_B: List[dict]):
    history_1, user_query_1, model_output_1 = parse_conversation(conversation_A)
    history_2, user_query_2, model_output_2 = parse_conversation(conversation_B)
    assert history_1 == history_2, "The history should be the same for both conversations"
    assert user_query_1 == user_query_2, "The user query should be the same for both conversations"
    return history_1, user_query_1, model_output_1, model_output_2

class PairwiseRMFeedback(BaseRMTemplate):
    template_path = os.path.join(os.path.dirname(__file__), "pairwise_feedback.template")
    def __init__(self, template_path=None):
        super().__init__(template_path)

    def apply_template(self, feedback: Union[str, dict]):
        return super().apply_template(feedback)
        

class PairwiseRMShot(BaseRMTemplate):
    template_path = os.path.join(os.path.dirname(__file__), "pairwise_shot.template")
    def __init__(self, template_path=None):
        super().__init__(template_path)
        
    def apply_template(self, index:int, conversation_A: List[dict], conversation_B: List[dict], feedback: Union[str, dict]):
        history, user_query, model_output_1, model_output_2 = parse_conversation_pair(conversation_A, conversation_B)
        template_keys = {
            "index": index,
            "history": history or "N/A",
            "user_query": user_query,
            "candidate_A": model_output_1,
            "candidate_B": model_output_2,
            "feedback": feedback
        }
        return super().apply_template(**template_keys)

class PairwiseRMTemplate(BaseRMTemplate):
    template_path = os.path.join(os.path.dirname(__file__), "pairwise_overall.template")
    def __init__(self, template_path=None, shot_template_path=None, feedback_template_path=None):

        self.shot_template = PairwiseRMShot(shot_template_path)
        self.feedback_template = PairwiseRMFeedback(feedback_template_path)
        super().__init__(template_path)

    def apply_template(self, conversation_A: List[dict], conversation_B: List[dict], shots: List[dict]):
        shots_strs = [self.shot_template.apply_template(i+1, shot["conversation_A"], shot["conversation_B"], shot["feedback"]) for i, shot in enumerate(shots)]
        shots_str = "\n".join(shots_strs).strip('\n ')
        
        history, user_query, model_outpu_1, model_output_2 = parse_conversation_pair(conversation_A, conversation_B)

        template_keys = {
            "history": history or "N/A",
            "user_query": user_query,
            "candidate_A": model_outpu_1,
            "candidate_B": model_output_2,
            "shots": shots_str
        }

        return super().apply_template(**template_keys)
    

if __name__ == "__main__":
    conversation_A = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm good, how can I help you?"},
        {"role": "user", "content": "I want to book a flight"},
        {"role": "assistant", "content": "Sure, where do you want to go?"}
    ]
    conversation_B = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm good, how can I help you?"},
        {"role": "user", "content": "I want to book a flight"},
        {"role": "assistant", "content": "Sure, where do you plan to go?"}
    ]
    shots = [
        {
            "conversation_A": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm good, how can I help you?"},
                {"role": "user", "content": "I want to book a flight"},
                {"role": "assistant", "content": "Sure, where do you want to go?"}
            ],
            "conversation_B": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm good, how can I help you?"},
                {"role": "user", "content": "I want to book a flight"},
                {"role": "assistant", "content": "Sure, where do you want to go?"}
            ],
            "feedback": "A is better"
        }
    ]

    template = PairwiseRMTemplate()
    print(template.apply_template(conversation_A, conversation_B, shots*4))