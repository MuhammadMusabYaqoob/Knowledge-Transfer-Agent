from langchain_google_genai import GoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file for environment variables

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY must be set in .env file")

llm = GoogleGenerativeAI(model="gemini-2.0-flash")

def get_llm():
    return llm