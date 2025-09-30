from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
import docx
import pandas as pd
import os
import chromadb
import ollama
from utils import get_collection, load_config
import xlwings as xw

config = load_config()

  
# Function to extract text from PDF files
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Function to extract text from DOCX files
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = "\n".join(para.text for para in doc.paragraphs)
    return text

# Function to extract text from TXT files
def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

# Function to extract text from XLSX files
def extract_text_from_xlsx(file_path):
    all_text = []
    app = xw.App(visible=False)
    try:
        wb = app.books.open(file_path)
        for sheet in wb.sheets:
            used_range = sheet.used_range
            if used_range.value:
                for row in used_range.value:
                    if row:
                        all_text.extend([str(cell) for cell in row if cell is not None])

            for shape in sheet.api.Shapes:
                try:
                    text = shape.TextFrame2.TextRange.Text
                    if text:
                        all_text.append(text)
                except Exception:
                    continue

        return "\n".join(all_text)

    finally:
        wb.close()
        app.quit()

# Function to split text into chunks using RecursiveCharacterTextSplitter
def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunks"]["chunk_size"],
        chunk_overlap=config["chunks"]["chunk_overlap"]
    )
    return splitter.split_text(text)

emb = config["embedding_model"]

# Function to extract, split and embed text from a tender folder
def process_tender(path):
    collection = get_collection()

    i = 0
    for entry in os.scandir(path):
        print(f"Extracting text from {entry}")
        ext = os.path.splitext(entry)[1].lower()
        if ext not in (".pdf", ".docx", ".xls", ".xlsx", ".txt"):
            continue
        try:
            if ext == '.pdf':
                text = extract_text_from_pdf(entry.path)
            elif ext == '.docx':
                text = extract_text_from_docx(entry.path)
            elif ext in ['.xls', '.xlsx']:
                text = extract_text_from_xlsx(entry.path)
            elif ext == '.txt':
                text = extract_text_from_txt(entry.path)
            else:
                return None
        except Exception as e:
            continue
        
        chunk = split_text_into_chunks(text)
        print(f"Splitted into {len(chunk)} chunks")

        for sp in chunk:
            response = ollama.embed(model=emb, input=str(sp))
            embeddings = response["embeddings"]
            i += 1
            collection.add(
                documents=[str(sp)],
                embeddings=embeddings,
                ids=[f"id{i}"],
            )
    print(f"{collection.count()} Embeddings added to collection")

def emb_c():
    collection = get_collection()
    return collection.count()


