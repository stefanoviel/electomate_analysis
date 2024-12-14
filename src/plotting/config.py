import os
from dotenv import load_dotenv
import openai

# Disable parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Load environment variables from .env file
load_dotenv(dotenv_path='Backend/Evals/.env')

# Initialize OpenAI client with API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=api_key)

# Model specifications and cutoffs
modelspec = "gpt-4o"  # gpt-4, gpt-4o, gpt-4o-mini
cutoff_questions = 0
cutoff_parties = 0