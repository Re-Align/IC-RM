import regex
import json5

def parse_json(json_string):
    """
    Parse a JSON string and return the corresponding Python object.
    
    Args:
        json_string (str): A string in JSON format.
        
    Returns:
        dict: The parsed JSON object.
    """
    start_idx = json_string.find('{')
    end_idx = json_string.rfind('}') + 1
    json_string = json_string[start_idx:end_idx]
    return json5.loads(json_string)