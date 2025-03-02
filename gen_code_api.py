import json
import requests
import os
from google import genai
from google.genai import types

API_ENDPOINTS = {
    "gemini": lambda prompt: gemini_gen(prompt),
    "deepseek": lambda prompt: deepseek_gen(prompt),
}

# read from $HOME/.key/gemini.api.key and set to gemini_api_key
with open(os.path.expanduser("~/.keys/gemini.api.key"), "r") as f:
    gemini_api_key = f.read().strip()

gemini_client = genai.Client(api_key=gemini_api_key)

def gemini_gen(messages):
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        # model="gemini-2.0-pro-exp-02-05",
        config=types.GenerateContentConfig(
            # response_mime_type="text/markdown",
            # system_instruction="",
        ),
        contents=messages, # TODO check here: https://ai.google.dev/gemini-api/docs/quickstart?lang=rest
    )
    return response.text


# read from $HOME/.key/fireworks.api.key and set to fireworks_api_key
with open(os.path.expanduser("~/.keys/fireworks.api.key"), "r") as f:
    fireworks_api_key = f.read().strip()
fireworks_endpoint = "https://api.fireworks.ai/inference/v1/chat/completions"
fireworks_model = "accounts/fireworks/models/deepseek-r1"

# read from $HOME/.key/deepseek.api.key and set to deepseek_api_key
with open(os.path.expanduser("~/.keys/deepseek.api.key"), "r") as f:
    deepseek_api_key = f.read().strip()
deepseek_endpoint = "https://api.deepseek.com/chat/completions"
deepseek_model = "deepseek-reasoner"

with open(os.path.expanduser("~/.keys/friendli.api.key"), "r") as f:
    friendly_api_key = f.read().strip()
friendly_endpoint = "https://api.friendli.ai/serverless/v1/chat/completions"
friendly_model = "deepseek-r1"


deepseek_provider = {
    "url": fireworks_endpoint,
    "model": fireworks_model,
    "api_key": fireworks_api_key,
}

deepseek_provider = {
    "url": friendly_endpoint,
    "model": friendly_model,
    "api_key": friendly_api_key,
}

# deepseek_provider = {
#     "url": deepseek_endpoint,
#     "model": deepseek_model,
#     "api_key": deepseek_api_key,
# }

def deepseek_gen(messages):
    url = deepseek_provider["url"]
    payload = {
        "model": deepseek_provider["model"],
        # "max_tokens": 8192,
        # "max_tokens": 20480,
        "max_tokens": 40960,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
        "messages": messages
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {deepseek_provider['api_key']}"
    }
    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    print(response)
    # print(json.dumps(response.json(), indent=4))
    return response.json()["choices"][0]["message"]

if __name__ == "__main__":
    from prompts import COMMON_PROMPT, MATMUL_PROMPT
    print(API_ENDPOINTS["gemini"](COMMON_PROMPT + MATMUL_PROMPT))
    print(API_ENDPOINTS["deepseek"](COMMON_PROMPT + MATMUL_PROMPT))
