import streamlit as st
from streamlit_option_menu import option_menu
import os
import shutil
import json
from extractor import process_tender
from query_db import get_json_answer, get_json_category
from utils import get_collection, load_config, get_prompt_path, get_query_path
from extractor import extract_text_from_txt

st.set_page_config(
    page_title="Tender Classification and Extraction",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.image(
        "https://img.icons8.com/document",
        width=50,
    )
    st.title("Tender Classification and Extraction")
    selected = option_menu(
        menu_title="Navigation",
        options=["Upload Tender", "About"],
        icons=["cloud-upload", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

config = load_config()
collection = get_collection(config["db_path"], config["collection_name"])
PATH = "./data"

if os.path.exists(PATH):
    shutil.rmtree(PATH)
os.makedirs(PATH)

if selected == "Upload Tender":
    st.header("Tender Classification")
    st.write("Upload a tender to classify and extract information.")

    uploaded_files = st.file_uploader(
        "Choose a PDF tender file",
        type=["pdf", "doc", "docx", "txt", "xls", "xlsx"],
        accept_multiple_files=True,
        help="Upload a tender file in PDF format.",
    )

    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(PATH, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"Uploaded ({len(uploaded_files)}) files successfully!")

        if st.button("Start Processing"):
            st.info("Processing")
            process_tender(PATH)

            def run_extraction_task():
                results = {}

                issuer_input = extract_text_from_txt(get_query_path(config, "issuer"))
                issuer_prompt = extract_text_from_txt(get_prompt_path(config, "issuer"))
                results["Issuer"] = get_json_answer(PATH, issuer_input, issuer_prompt, 5)

                description_input = extract_text_from_txt(get_query_path(config, "description"))
                description_prompt = extract_text_from_txt(get_prompt_path(config, "description"))
                results["Service Description"] = get_json_answer(PATH, description_input, description_prompt, 5)

                bidder_input = extract_text_from_txt(get_query_path(config, "bidder"))
                bidder_prompt = extract_text_from_txt(get_prompt_path(config, "bidder"))
                results["Bidder Profile"] = get_json_answer(PATH, bidder_input, bidder_prompt, 8)

                category_prompt = extract_text_from_txt(get_prompt_path(config, "category"))
                results["Category"] = get_json_category(results["Service Description"], category_prompt)

                return results
            
            output = run_extraction_task()
            st.success("Extraction completed!")
            st.subheader("Extracted Results")
            st.json(output)

if selected == "About":
    st.header("About This App")
    st.markdown(""" """)          