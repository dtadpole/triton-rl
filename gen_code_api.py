import json
import requests
import os
import datetime
import asyncio
import aiohttp
from google import genai
from google.genai import types

# read from $HOME/.key/gemini.api.key and set to gemini_api_key
with open(os.path.expanduser("~/.keys/gemini.api.key"), "r") as f:
    gemini_api_key = f.read().strip()

gemini_client = genai.Client(api_key=gemini_api_key)

async def gemini_gen(messages, log_folder="."):
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

# deepseek_provider = {
#    "url": friendly_endpoint,
#    "model": friendly_model,
#    "api_key": friendly_api_key,
# }

# deepseek_provider = {
#     "url": deepseek_endpoint,
#     "model": deepseek_model,
#     "api_key": deepseek_api_key,
# }

async def post_data(url, data, headers):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                print("Error: ", response.status)
                print(await response.text())
                # throw an error
                raise Exception("Error: ", response)

async def deepseek_gen(messages, log_folder="."):
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

    # write request_messages to log file
    with open(os.path.join(log_folder, "api_request.json"), "w") as f:
        f.write(json.dumps(messages, indent=4))

    # write request message content to log file
    with open(os.path.join(log_folder, "api_request.md"), "w") as f:
        # parse messages[0]['content']
        json_content = json.loads(messages[0]['content'])
        for content in json_content:
            f.write(f"# {content['type']}")
            f.write("\n")
            if content['type'] == 'file':
                f.write(f"```{content['filetype']}\n")
                f.write(content['text'])
                f.write("```)\n")
            else:
                f.write(content['text'])
            f.write("\n\n")

    # response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    response = await post_data(url, payload, headers)
    response = await post_data(url, payload, headers)

    # write response to log file
    with open(os.path.join(log_folder, "api_response.json"), "w") as f:
        f.write(json.dumps(response, indent=4))

    with open(os.path.join(log_folder, "api_response.md"), "w") as f:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        f.write(f"Timestamp: {timestamp}\n\n")

        if "content" in response["choices"][0]["message"]:
            # if response_message has a "reasoner-content" key, write it to the log file
            if "reasoning_content" in response["choices"][0]["message"] and response["choices"][0]["message"]["reasoning_content"]:
                f.write("<think>\n")
                f.write(response["choices"][0]["message"]["reasoning_content"])
                f.write("\n</think>\n")

            f.write(response["choices"][0]["message"]["content"])
        else:
            f.write(json.dumps(response["choices"][0]["message"], indent=4))

    return response["choices"][0]["message"]["content"]


with open(os.path.expanduser("~/.keys/claude.api.key"), "r") as f:
    claude_api_key = f.read().strip()

async def claude_gen(messages, log_folder="."):
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 40960,
        "thinking": {
            "type": "enabled",
            "budget_tokens": 20480,
        },
        "messages": messages,
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": claude_api_key,
    }

    # write request_messages to log file
    with open(os.path.join(log_folder, "api_request.json"), "w") as f:
        f.write(json.dumps(messages, indent=4))

    # write request message content to log file
    with open(os.path.join(log_folder, "api_request.md"), "w") as f:
        # iterate through json content
        for message in messages:
            f.write(f"# {message['role']}")
            f.write("\n")
            # parse json content
            json_content = json.loads(message['content'])
            for content in json_content:
                f.write(f"# {content['type']}")
                f.write("\n")
                if content['type'] == 'file':
                    f.write(f"```{content['filetype']}\n")
                    f.write(content['text'])
                    f.write("```\n")
                else:
                    f.write(content['text'])
                f.write("\n\n")

    # response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    response = await post_data(url, payload, headers)

    # write response to log file
    with open(os.path.join(log_folder, "api_response.json"), "w") as f:
        f.write(json.dumps(response, indent=4))

    with open(os.path.join(log_folder, "api_response.md"), "w") as f:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        f.write(f"Timestamp: {timestamp}\n\n")

        # iterate through json content
        result = ""
        for content in response["content"]:
            # iterate key value pairs
            f.write(f"# {content['type']}")
            f.write("\n")
            if content['type'] == 'thinking':
                f.write(content['thinking'])
            elif content['type'] == 'text':
                f.write(content['text'])
                result += content['text']
            f.write("\n\n")

    return result

API_ENDPOINTS = {
    "gemini": gemini_gen,
    "deepseek": deepseek_gen,
    "claude": claude_gen,
}

if __name__ == "__main__":
    from prompts import COMMON_PROMPT, MATMUL_PROMPT
    print(API_ENDPOINTS["gemini"](COMMON_PROMPT + MATMUL_PROMPT))
    print(API_ENDPOINTS["deepseek"](COMMON_PROMPT + MATMUL_PROMPT))
