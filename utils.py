import chromadb
import json
import os

def get_client(db_path="chroma_db"):
    return chromadb.PersistentClient(path=db_path)

def get_collection(db_path="chroma_db", collection_name="tender_processing"):
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name=collection_name)

def load_config(path="config.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_prompt_path(config, filename_key):
    return os.path.join(config["input_dir"], config["prompts"][filename_key])

def get_query_path(config, filename_key):
    return os.path.join(config["input_dir"], config["queries"][filename_key])