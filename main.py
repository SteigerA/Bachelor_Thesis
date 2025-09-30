import chromadb
import json
from extractor import extract_text_from_txt, process_tender
from query_db import get_json_answer, get_json_category
from utils import get_collection, load_config, get_prompt_path, get_query_path, get_client
import ollama

config = load_config()
client = get_client(config["db_path"])
collection = get_collection(config["db_path"], config["collection_name"])


PATH = "./data"


process_tender(PATH)


output_json_file = f"extracted_results.json" 

# Create an empty JSON file
with open(output_json_file, 'w', encoding='utf-8') as f:
    json.dump({}, f, indent=4, ensure_ascii=False)


# Function to safely update JSON file
def safe_json(new_data):
    # Load JSON
    with open(output_json_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {}

    data.update(new_data)
    
    # Save updated JSON
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Output saved as {output_json_file}")


# Search for issuer
issuer_input = extract_text_from_txt(get_query_path(config, "issuer"))
issuer_prompt = extract_text_from_txt(get_prompt_path(config, "issuer"))
i_answer = get_json_answer(PATH, issuer_input, issuer_prompt, 5)
safe_json(i_answer)

# Search for description
description_input = extract_text_from_txt(get_query_path(config, "description"))
description_prompt = extract_text_from_txt(get_prompt_path(config, "description"))
d_answer = get_json_answer(PATH, description_input, description_prompt, 5)
safe_json(d_answer)

# Search for deadline
submission_input = extract_text_from_txt(get_query_path(config, "submission"))
submission_prompt = extract_text_from_txt(get_prompt_path(config, "submission"))
s_answer = get_json_answer(PATH, submission_input, submission_prompt, 10)
safe_json(s_answer)

# Search for deadline
duration_input = extract_text_from_txt(get_query_path(config, "duration"))
duration_prompt = extract_text_from_txt(get_prompt_path(config, "duration"))
du_answer = get_json_answer(PATH, duration_input, duration_prompt, 10)
safe_json(du_answer)

# Categorize tender
category_prompt = extract_text_from_txt(get_prompt_path(config, "category"))
c_answer = get_json_category(d_answer, category_prompt)
safe_json(c_answer)


# delete collection after use
client.delete_collection(name=config["collection_name"])