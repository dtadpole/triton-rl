fireworks_api_key = "fw_3ZLEAizPSPuTZhKnoKdf55Ri"

import requests
import json

def deepseek_gen(prompt):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    payload = {
        "model": "accounts/fireworks/models/deepseek-r1",
        "max_tokens": 20480,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
        "messages": [
            {
                "role": "system",
                "content": "",
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {fireworks_api_key}"
    }
    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    # print(json.dumps(response.json(), indent=4))
    return response.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    from prompts import COMMON_PROMPT, MATMUL_PROMPT
    print(deepseek_gen(COMMON_PROMPT + MATMUL_PROMPT))
