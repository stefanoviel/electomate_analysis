import os
from dotenv import load_dotenv

# Disable parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Load environment variables from .env file
load_dotenv(dotenv_path='Backend/Evals/.env')

# Model specifications and cutoffs
MODEL_SPEC = "gpt-4o"  # gpt-4, gpt-4o, gpt-4o-mini
CUTOFF_QUESTIONS = 0
CUTOFF_PARTIES = 12

# OpenAI configuration
API_KEY = os.getenv("OPENAI_API_KEY")