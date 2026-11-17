import os
from google import genai

api_key = os.environ.get("GEMINI_API_KEY", "").strip()
if not api_key:
    raise RuntimeError("Set GEMINI_API_KEY before listing models.")
client = genai.Client(api_key=api_key)

models = client.models.list()
for m in models:
    print(m.name)