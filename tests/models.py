from google import genai

client = genai.Client(api_key="AIzaSyBzV-kWPffJ7-DGiqSAJdhY4n70u1JCicQ")

models = client.models.list()
for m in models:
    print(m.name)