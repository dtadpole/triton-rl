from google import genai
from google.genai import types

SYS_PROMPT="""
Respond in the following format, start with a "think" section, followed by one or more code blocks in markdown, and followed by a "test" section.
In the "think" section, give comprehensive, logically structured thinking, use markdown formatting for clear information organization, admit uncertainties for ambiguous situations.
In the code blocks, each block represent one python file. The first line of the code block is always a comment with filename, and represnet the runnable code in a file.
In the test section, generate instructions and commands to test the code.
<think>
...
</think>
```python
# filename: file_1.py
...
```
```python
# filename: file_2.py
...
```
```python
# filename: file_x.py
...
```
<test>
...
</test>
"""

client = genai.Client(api_key="AIzaSyBk5082INxetg6K79MFycnzmiRMFAwTjLQ")

response = client.models.generate_content(
    # model="gemini-2.0-flash",
    model="gemini-2.0-pro-exp-02-05",
    config=types.GenerateContentConfig(
        # response_mime_type="text/markdown",
        system_instruction=SYS_PROMPT,
    ),
    contents="generate triton code for RMSNorm, compare the performance with pytorch",
)

print(response.text)

# match response.text with regex and extract the code blocks
import re

code_blocks = re.findall(r'```python\n(.*?)```', response.text, re.DOTALL)

# print(code_blocks)

# write the code blocks to files
for code_block in code_blocks:
    filename = code_block.split("\n")[0].split("# filename: ")[1]
    with open(filename, "w") as f:
        f.write(code_block)
