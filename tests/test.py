from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key="AIzaSyBzV-kWPffJ7-DGiqSAJdhY4n70u1JCicQ")

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents="Tell me who messi is"
)
print(response.text)