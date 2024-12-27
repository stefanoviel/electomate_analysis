import requests
from llama_index.core import VectorStoreIndex, Document, ServiceContext
from llama_index.core.storage.storage_context import StorageContext
from pathlib import Path
import os
from PyPDF2 import PdfReader

party_manifestos = {
    "FDP": "https://www.fdp.de/sites/default/files/2024-03/2024-01-28_ept_das-programm-der-fdp-zur-europawahl-2024-1-_0.pdf",
    "CDU": "https://www.europawahl.cdu.de/sites/www.europawahlprogramm.cdu.de/files/docs/europawahlprogramm-cdu-csu-2024_0.pdf",
    # Add other parties...
}

def download_pdf(url, party_name):
    response = requests.get(url)
    pdf_path = f"manifestos/{party_name}_manifesto.pdf"
    os.makedirs("manifestos", exist_ok=True)
    
    with open(pdf_path, "wb") as f:
        f.write(response.content)
    return pdf_path


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Create a directory for storing indices
PERSIST_DIR = "./stored_indices"
party_indices = {}

def load_or_create_indices():
    for party, url in party_manifestos.items():
        party_persist_dir = os.path.join(PERSIST_DIR, party)
        
        # Check if index already exists
        if os.path.exists(os.path.join(party_persist_dir, "docstore.json")):  # Check for actual index file
            # Load existing index
            storage_context = StorageContext.from_defaults(persist_dir=party_persist_dir)
            index = VectorStoreIndex.load(storage_context=storage_context)
        else:
            # Create new index
            pdf_path = download_pdf(url, party)
            text = extract_text_from_pdf(pdf_path)
            documents = [Document(text=text)]
            
            # Create and save index
            os.makedirs(party_persist_dir, exist_ok=True)
            index = VectorStoreIndex.from_documents(documents)  # Create index first
            
            # Then set up storage context and persist
            storage_context = StorageContext.from_defaults(persist_dir=party_persist_dir)
            index.set_storage_context(storage_context)
            index.storage_context.persist()
            
        party_indices[party] = index

def query_party_manifesto(party_name, query):
    if party_name in party_indices:
        query_engine = party_indices[party_name].as_query_engine()
        response = query_engine.query(query)
        return response
    else:
        return f"Party {party_name} not found"

# Load or create indices when script starts
load_or_create_indices()

# Example usage
query = "What is the party's position on climate change?"
party = "FDP"
response = query_party_manifesto(party, query)
#print(f"{party}'s response:", response)