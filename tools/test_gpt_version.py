#Calls the model with a probe prompt and gets its version string from the metadata of its response.
#Can be used to freeze the used version for later reproducability.
#Website to compare models: https://developers.openai.com/api/docs/models/compare
#As of 19/05/2026:
#"gpt-4" returns "gpt-4-0613" - Legacy model used originally, not recommended to use due to high cost.
#"gpt-4o" returns "gpt-4o-2024-08-06"
#"gpt-5" returns "gpt-5-2025-08-07"
#"gpt-5.4" returns "gpt-5.4-2026-03-05" - Model used for S2 BSP
from pathlib import Path
from openai import OpenAI

model = input("Enter model: ")
endpoint = "https://api.openai.com/v1"


def read_api_key():
    base_dir = Path(__file__).resolve().parent.parent
    token_path = base_dir / "misc" / "api-key.txt"
    
    with open(token_path, "r") as file:
        mykey = file.read()
    return mykey

api_key = read_api_key()

client = OpenAI(api_key = api_key, base_url = endpoint)

response = client.chat.completions.create(
    model = model,
    messages=[
        {"role": "user", "content": "Hi"}
    ]
)

print("--------------")
print("Full model version:", response.model)
print("--------------")
print("Full response:")
print(response)