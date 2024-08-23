import regex
import json5

def extract_json_key_str(json_string):
    extracted_json = {}
    for line in json_string.split('\n'):
        line = line.strip()
        first_colon = line.find(':')
        key = line[:first_colon].strip().strip('"')
        value = line[first_colon+1:].strip().strip('"')
        if key and value:
            extracted_json[key] = value
    return extracted_json

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
        try:
            extracted_json = extract_json_key_str(json_string)
            return extracted_json
        except Exception as e:
            pass
        print(f"Error parsing JSON: {e}")
        print(json_string)
        return json_string