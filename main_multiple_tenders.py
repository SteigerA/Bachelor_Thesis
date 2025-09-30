import chromadb
import json
from extractor import extract_text_from_txt, process_tender, emb_c
from query_db import get_json_answer, get_json_category
from utils import get_collection, load_config, get_prompt_path, get_query_path, get_client
import ollama
import os
import extractor
import query_db  
import time 

config = load_config()
client = get_client(config["db_path"])

def safe_json(file_name, new_data):
    with open(file_name, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {}

    data.update(new_data)
    
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Output saved as {file_name}")

def process_single_tender(PATH, number):
    start_time = time.time()
    
    # Process documents
    process_tender(PATH)
    emb_num = emb_c()
    
    # Process all fields
    issuer_input = extract_text_from_txt(get_query_path(config, "issuer"))
    issuer_prompt = extract_text_from_txt(get_prompt_path(config, "issuer"))
    i_answer = get_json_answer(PATH, issuer_input, issuer_prompt, 5)
    
    description_input = extract_text_from_txt(get_query_path(config, "description"))
    description_prompt = extract_text_from_txt(get_prompt_path(config, "description"))
    d_answer = get_json_answer(PATH, description_input, description_prompt, 5)
    
    submission_input = extract_text_from_txt(get_query_path(config, "submission"))
    submission_prompt = extract_text_from_txt(get_prompt_path(config, "submission"))
    s_answer = get_json_answer(PATH, submission_input, submission_prompt, 10)
    
    duration_input = extract_text_from_txt(get_query_path(config, "duration"))
    duration_prompt = extract_text_from_txt(get_prompt_path(config, "duration"))
    du_answer = get_json_answer(PATH, duration_input, duration_prompt, 10)
    
    # c_answer = get_json_category(d_answer, extract_text_from_txt(get_prompt_path(config, "category")))
    
    # Calculate processing time
    processing_time = round(time.time() - start_time, 2)
    
    # Combine all data including metrics
    final_data = {
        **i_answer,
        **d_answer,
        **s_answer,
        **du_answer,
        # **c_answer,
        "_metrics": {
            "processing_time_sec": processing_time,
            "embedding_count": emb_num
        }
    }
    
    safe_json(number, final_data)

for entry in os.scandir("./Test_data"):
    name = os.path.basename(entry)
    path = f"./Test_data/{name}"
    output_json_file = f"{name}.json"

    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump({}, f, indent=4, ensure_ascii=False)

    collection = get_collection()
    extractor.collection = collection
    query_db.collection = collection

    process_single_tender(path, output_json_file)
    
    # delete collection after use
    client.delete_collection(name="tender_processing")
