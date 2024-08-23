import regex
import json5

def parse_to_json(json_string):
    """
    Parse a JSON string and return the corresponding Python object.
    
    Args:
        json_string (str): A string in JSON format.
        
    Returns:
        dict: The parsed JSON object.
    """
    if not isinstance(json_string, str):
        return json_string
    if not json_string:
        return {}
    try:
        start_idx = json_string.find('{')
        end_idx = json_string.rfind('}') + 1
        json_string = json_string[start_idx:end_idx]
        return json5.loads(json_string)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return json_string