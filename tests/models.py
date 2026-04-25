from google import genai

client = genai.Client(api_key="REMOVED_REVOKED_CREDENTIAL")

models = client.models.list()
for m in models:
    print(m.name)