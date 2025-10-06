from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file for environment variables

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")

supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase():
    return supabase_client