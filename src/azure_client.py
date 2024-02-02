import os

from dotenv import load_dotenv
from openai import AzureOpenAI

def initialize_client() -> AzureOpenAI:
    load_dotenv(dotenv_path=".environment")
    return AzureOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("OPENAI_API_BASE"),
    )
