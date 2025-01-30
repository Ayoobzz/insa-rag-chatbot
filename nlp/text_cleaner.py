import json

char_mapping = {
    "é": "e", "è": "e", "ê": "e", "ë": "e",
    "à": "a", "â": "a", "ä": "a",
    "î": "i", "ï": "i",
    "ô": "o", "ö": "o",
    "ù": "u", "û": "u", "ü": "u",
    "ç": "c",
    "ÿ": "y"
}

def replace_unsupported_chars(text):
    """Replace unsupported French letters in a given text."""
    if isinstance(text, str):
        for char, replacement in char_mapping.items():
            text = text.replace(char, replacement)
    return text

def process_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)


    def recursive_replace(obj):
        if isinstance(obj, dict):
            return {key: recursive_replace(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [recursive_replace(item) for item in obj]
        elif isinstance(obj, str):
            return replace_unsupported_chars(obj)
        return obj

    processed_data = recursive_replace(data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)


input_json = "../data/results.json" # Replace with your input file name
output_json = "../data/cleaned.json" # Replace with your desired output file name

process_json(input_json, output_json)
print(f"Processed JSON saved to {output_json}")
