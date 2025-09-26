from tavily import TavilyClient
import os

# Load API key from environment to avoid hardcoding secrets
tavily_key = os.getenv("TAVILY_API_KEY")
if not tavily_key:
    raise SystemExit("TAVILY_API_KEY not set in environment")

tavily_client = TavilyClient(api_key=tavily_key)
response = tavily_client.search("Who is Leo Messi?")

print(response)
