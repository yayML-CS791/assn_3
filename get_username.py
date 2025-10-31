hf_token = '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'

import requests

token = "hf_your_token_here"  # Replace with your actual token
headers = {"Authorization": f"Bearer {hf_token}"}
response = requests.get("https://huggingface.co/api/whoami-v2", headers=headers)

if response.status_code == 200:
    user_info = response.json()
    print(f"Username: {user_info['name']}")
    print(f"Full details: {user_info}")
else:
    print(f"Error: {response.status_code} - {response.text}")