import os
import datetime
import subprocess
from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyBk5082INxetg6K79MFycnzmiRMFAwTjLQ")

def gemini_gen(prompt):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        # model="gemini-2.0-pro-exp-02-05",
        config=types.GenerateContentConfig(
            # response_mime_type="text/markdown",
            # system_instruction="",
        ),
        contents=[prompt],
    )
    return response.text


if __name__ == "__main__":
    from prompts import COMMON_PROMPT, MATMUL_PROMPT
    print(gemini_gen(COMMON_PROMPT + MATMUL_PROMPT))
