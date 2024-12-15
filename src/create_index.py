from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from tqdm import tqdm

print("Loading documents...")
# Add tqdm to show progress while loading documents
documents = SimpleDirectoryReader("downloaded_pdfs").load_data()

print("Creating index...")
# Wrap the documents with tqdm to show progress while indexing
index = VectorStoreIndex.from_documents(
    tqdm(documents, desc="Indexing documents", unit="doc")
)

print("Persisting index to disk...")
# Save the index
index.storage_context.persist(persist_dir="index_store")
print("Done!")
