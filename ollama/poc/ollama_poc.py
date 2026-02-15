import json
import urllib.request

def ollama_generate(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode('utf-8'))
        return result
    

# URL for local ollama server
url = "http://localhost:11434/api/generate"

prompt = "In one paragraph explaing what a Transformer attention head does."

# Create the payload for the POST request
payload = {
    "model": "deepseek-r1:8b",
    "prompt": prompt,
    "options": {
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "seed": 42
    },
    "stream": False
}

# Verify options
print("Payload options: ", payload["options"])

data = json.dumps(payload).encode('utf-8')

result = ollama_generate(url, payload)
result2 = ollama_generate(url, payload)

print(f"(tokens eval: {result.get('eval_count')}, seconds: {result.get('total_duration', 0)/1e9:.2f})")
print(result["response"])

print(f"(tokens eval: {result2.get('eval_count')}, seconds: {result2.get('total_duration', 0)/1e9:.2f})")
print(result2["response"])

print("\nOutputs identical:", result["response"] == result2["response"])
