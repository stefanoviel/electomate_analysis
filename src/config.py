import os
from dotenv import load_dotenv
import openai

# Load environment variables from the .env file located in the src directory
load_dotenv(dotenv_path='src/.env')

# Retrieve the API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=api_key)

# Check if the API key is loaded correctly
if not api_key:
    raise ValueError("API key not found. Please ensure the .env file is set up correctly.")

# Initialize the OpenAI client with the API key
openai.api_key = api_key

# Model specifications and cutoffs
modelspec = "gpt-4o"  # gpt-4, gpt-4o, gpt-4o-mini
cutoff_questions = 38
cutoff_parties = 1
is_rag_context = True

# New configuration options
disable_parallelization = True  # Set to True to disable parallel processing
chunk_size = 512  # Size of each chunk
chunk_overlap = 50  # Overlap between chunks