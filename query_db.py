import ollama
import json
import chromadb
from extractor import extract_text_from_txt
from utils import get_collection, load_config

config = load_config()
llm = config["llm_model"]
emb = config["embedding_model"]

def query_db(input, res):
    response = ollama.embed(
    model=emb,
    input=input
    )

    results = collection.query(
    query_embeddings=response["embeddings"],
    n_results=res
    )

    data = results['documents']
    return data

def ask_ollama(data, prompt):
    output = ollama.generate(
    model=llm,
    prompt=f"Using this data: {data}. Respond to this prompt: {prompt}."
    )
    return output['response']

def parse_json_safely(text):
    """Safely extract and parse JSON from text, handling potential errors."""
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            json_str = text[start : end + 1]
            parsed_json = json.loads(json_str)
            return parsed_json
        return {"error": "No JSON content found"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON content"}

def get_json_answer(PATH, input, prompt, res):
    data = query_db(input, res)
    answer = ask_ollama(data, prompt)
    json_format = parse_json_safely(answer)
    return json_format

def get_json_category(input, prompt):
    answer = ask_ollama(input, prompt)
    json_format = parse_json_safely(answer)
    return json_format


collection = get_collection()