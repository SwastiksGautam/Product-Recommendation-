import requests

MCP_SERVER_URL = "http://127.0.0.1:8000"
query = "mechanical engineer"
top_n = 5

resp = requests.get(f"{MCP_SERVER_URL}/semantic_search", params={"query": query, "top_n": top_n})
products = resp.json()

print(f"Returned {len(products)} products")
for p in products:
    print(p['name'])
