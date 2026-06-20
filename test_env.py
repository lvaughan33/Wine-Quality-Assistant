import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

print(api_key is not None)

if api_key:
    print(api_key[:12] + "...")