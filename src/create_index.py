from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SimpleNodeParser
from tqdm import tqdm
import os
from config import openai_client, modelspec, chunk_size, chunk_overlap

# Define chunk size parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

print("Loading documents...")
documents = SimpleDirectoryReader("downloaded_pdfs").load_data()



def generate_llm_based_metadata(documents):
    """
    Generate metadata for each document using GPT to identify party and document type
    from the first pages of each PDF.
    """
    system_prompt = (
        "You are a helpful assistant that classifies German political documents. "
        "Given text from a PDF, determine:\n"
        "1) The political party this document belongs to (in German).\n"
        "2) The type of document (e.g., 'manifesto', 'program', 'press release', etc.).\n"
        "Output your answer in JSON with keys 'party' and 'doc_type'."
    )

    for doc in tqdm(documents, desc="Generating document metadata"):
        # Extract first portion of text (roughly 2 pages worth)
        text_sample = doc.text[:4000]  # Adjust size as needed
        
        user_prompt = (
            "Here is text from the beginning of a German political document:\n\n"
            f"{text_sample}\n\n"
            "Please identify:\n"
            "- which German political party this document belongs to\n"
            "- the type of document\n"
            "Return JSON: { \"party\": \"...\", \"doc_type\": \"...\" }"
        )

        try:
            response = openai_client.chat.completions.create(
                model=modelspec,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            content = response.choices[0].message.content
            
            # Parse LLM response
            import json
            metadata_dict = json.loads(content)
            
            # Attach metadata to document
            doc.metadata["party"] = metadata_dict.get("party", "Unknown")
            doc.metadata["doc_type"] = metadata_dict.get("doc_type", "Unknown")
            
            print(f"Identified: {metadata_dict}")
            
        except Exception as e:
            print(f"Error processing document: {e}")
            doc.metadata["party"] = "Unknown"
            doc.metadata["doc_type"] = "Unknown"

    return documents




# Generate metadata for documents
print("\nAnalyzing documents for metadata...")
documents = generate_llm_based_metadata(documents)

# Initialize response variable
response = 'o'  # default to overwrite, needed to not cause error

# Check for existing index
if os.path.exists("index_store"):
    response = input("Existing index found. Do you want to: [o]verwrite, [u]pdate, or [c]ancel? ")
    if response.lower() == 'c':
        print("Aborting...")
        exit()
    elif response.lower() == 'u':
        # Load existing index
        storage_context = StorageContext.from_defaults(persist_dir="index_store")
        index = load_index_from_storage(storage_context)
        existing_nodes = index.docstore.docs.values()
        existing_doc_ids = {node.metadata['file_name'] for node in existing_nodes if 'file_name' in node.metadata}
        
        # Filter out documents that are already indexed
        new_documents = [doc for doc in documents if doc.metadata['file_name'] not in existing_doc_ids]
        if not new_documents:
            print("No new documents to index.")
            exit()
        
        print(f"Found {len(new_documents)} new documents to index")
        documents = new_documents

# Create a node parser with custom chunk size
node_parser = SimpleNodeParser.from_defaults(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

# Parse documents into nodes (chunks) with progress bar
print("Generating chunks...")
nodes = []
for doc in tqdm(documents, desc="Processing documents", unit="doc"):
    doc_nodes = node_parser.get_nodes_from_documents([doc])
    # Propagate document metadata to chunks
    for node in doc_nodes:
        node.metadata["party"] = doc.metadata.get("party", "Unknown")
        node.metadata["doc_type"] = doc.metadata.get("doc_type", "Unknown")
    nodes.extend(doc_nodes)

print(f"Generated {len(nodes)} chunks in total")

print("Creating/Updating index...")
with tqdm(total=len(nodes), desc="Indexing chunks", unit="chunk") as pbar:
    if response.lower() == 'u':
        # Add new nodes to existing index
        index.insert_nodes(nodes)
    else:
        # Create new index
        index = VectorStoreIndex(nodes=nodes)
    pbar.update(len(nodes))

print("Persisting index to disk...")
index.storage_context.persist(persist_dir="index_store")
print("Done!")
