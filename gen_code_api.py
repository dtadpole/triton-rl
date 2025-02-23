import json
import requests
from google import genai
from google.genai import types

API_ENDPOINTS = {
    "gemini": lambda prompt: gemini_gen(prompt),
    "deepseek": lambda prompt: deepseek_gen(prompt),
}

benchmark_code = open("benchmark.py").read()


chutes_fingerprint = "x7EkUNSjOn6AIuVyqBGglespCtPHdMYF"

gemini_client = genai.Client(api_key="AIzaSyBk5082INxetg6K79MFycnzmiRMFAwTjLQ")

def gemini_gen(prompt):
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        # model="gemini-2.0-pro-exp-02-05",
        config=types.GenerateContentConfig(
            # response_mime_type="text/markdown",
            # system_instruction="",
        ),
        contents=[prompt],
    )
    return response.text



fireworks_api_key = "fw_3ZLEAizPSPuTZhKnoKdf55Ri"

def deepseek_gen(prompt):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    payload = {
        "model": "accounts/fireworks/models/deepseek-r1",
        # "max_tokens": 20480,
        "max_tokens": 40960,
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
            },
            {
                "role": "file",
                "name": "benchmark.py",
                "content": benchmark_code,
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
    from gen_code_prompts import COMMON_PROMPT, MATMUL_PROMPT
    print(API_ENDPOINTS["gemini"](COMMON_PROMPT + MATMUL_PROMPT))
    print(API_ENDPOINTS["deepseek"](COMMON_PROMPT + MATMUL_PROMPT))
