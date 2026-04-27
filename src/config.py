"""
config.py — Shared Gemini client and model name.
Imported by main.py, agents.py, benchmark.py — single source of truth.
"""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = "gemini-2.5-flash" 
