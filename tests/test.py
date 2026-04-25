from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(api_key="REMOVED_REVOKED_CREDENTIAL")

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents="Tell me who messi is"
)
print(response.text)