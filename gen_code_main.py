import os
import datetime
import subprocess
import json
import asyncio
import time
from google import genai
from google.genai import types
from gen_code_prompts import FUNC_GEN_PROMPT, FUNC_FORMAT_PROMPT, CODE_FIX_PROMPT
from gen_code_api import API_ENDPOINTS
import argparse

SUCCESS = False
TEST_COMMAND = ""


def get_folder_name(args):
    # get the next sequence id
    next_sequence_id = 0
    # check for next available subfolder name, starts with code_gen_, and ends with a number
    while os.path.exists(
        os.path.join(args.dir, f"code_gen_{args.api}_{next_sequence_id:03d}")
    ):
        next_sequence_id += 1
    folder_name = os.path.join(args.dir, f"code_gen_{args.api}_{next_sequence_id:03d}")
    os.makedirs(folder_name, exist_ok=True)
    # for files with name "prompt.txt"  or ending with ".py" or ".stderr" or ".stdout" or ".failed" or ".success" or ".command " in the args.folder, read the content and write to subfolder_name
    for file in os.listdir(args.dir):
        if (
            file == "prompt.txt"
            or file.endswith(".py")
            or file.endswith(".stderr")
            or file.endswith(".stdout")
            or file.endswith(".success")
            or file.endswith(".failed")
            or file.endswith(".command")
        ):
            with open(os.path.join(args.dir, file), "r") as f:
                content = f.read()
            with open(os.path.join(folder_name, file), "w") as f:
                f.write(content)
    return folder_name


# queue is a list of parent and child folders
async def initialize_queue(args, folder_name, queue):
    # find start_iteration by checking folder names in folder_name
    start_iteration = 0
    for _ in range(args.breadth):
        while os.path.exists(os.path.join(folder_name, f"{start_iteration:02d}")):
            start_iteration += 1
        # append start_iteration to queue
        child_config = {
            "parent": folder_name,
            "child": os.path.join(folder_name, f"{start_iteration:02d}"),
            "depth": 0,
        }
        print(f"Adding {child_config} to queue")
        await queue.put(child_config)
        start_iteration += 1


def codegen_messages(args, folder_name):
    INPUT_PROMPT = open(os.path.join(args.dir, "prompt.txt")).read()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": FUNC_GEN_PROMPT},
                {"type": "text", "text": FUNC_FORMAT_PROMPT},
                {"type": "text", "text": INPUT_PROMPT},
            ],
        },
    ]
    # list folder with all files ending with .py
    files = [f for f in os.listdir(folder_name) if f.endswith(".py")]
    for file in files:
        with open(os.path.join(folder_name, file), "r") as f:
            messages[0]["content"].append(
                {
                    "type": "file",
                    "filename": file,
                    "filetype": "python",
                    "text": f.read(),
                }
            )
    # return messages
    return messages


def fix_messages(args, folder_name, err_message):
    messages = codegen_messages(args, folder_name)
    messages[0]["content"].append({"type": "text", "text": CODE_FIX_PROMPT})
    messages[0]["content"].append(
        {"type": "text", "text": f"Error Message: {err_message}"}
    )
    return messages


async def run_command_async(cmd):
    """Run a shell command asynchronously and capture its output."""
    start_time = time.time()

    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=os.environ,
    )
    stdout, stderr = await process.communicate()

    end_time = time.time()
    duration = end_time - start_time

    return {
        "cmd": cmd,
        "stdout": stdout.decode("utf-8"),
        "stderr": stderr.decode("utf-8"),
        "returncode": process.returncode,
        "duration": duration,
    }


async def processor(args, queue, id, timeout=1):

    global SUCCESS
    global TEST_COMMAND

    print("--------------------------------")
    print(f"Processor {id} started")

    exception_count = 0
    # while not SUCCESS
    while not SUCCESS:
        # pop the first item from queue
        print("--------------------------------")
        try:
            iter_config = await asyncio.wait_for(queue.get(), timeout)
            # add the current folder to queue
            print(f"Processing {iter_config}")
            print("--------------------------------")
            if iter_config["depth"] < args.depth:
                iter_count = 0
                for _ in range(args.breadth):
                    while os.path.exists(
                        os.path.join(iter_config["child"], f"{iter_count:02d}")
                    ):
                        iter_count += 1
                    # append start_iteration to queue
                    child_config = {
                        "parent": iter_config["child"],
                        "child": os.path.join(
                            iter_config["child"], f"{iter_count:02d}"
                        ),
                        "depth": iter_config["depth"] + 1,
                    }
                    print(f"Adding {child_config} to queue")
                    await queue.put(child_config)
                    iter_count += 1
                print("--------------------------------")
        except asyncio.TimeoutError:
            # check if queue is empty
            if queue.empty():
                break
            else:
                continue

        try:
            base_folder = iter_config["parent"]
            code_folder = iter_config["child"]

            # ERR_MESSAGE is the error message from the last iteration
            ERR_MESSAGE = ""
            if os.path.exists(os.path.join(base_folder, "code.stderr")):
                with open(os.path.join(base_folder, "code.stderr"), "r") as f:
                    ERR_MESSAGE = f.read()

            # if there is a .failed file, set FIX_MODE to True
            if os.path.exists(os.path.join(iter_config["parent"], ".failed")):
                FIX_MODE = True
                request_messages = fix_messages(
                    args, iter_config["parent"], ERR_MESSAGE
                )
            else:
                FIX_MODE = False
                request_messages = codegen_messages(args, iter_config["parent"])

            request_messages[0]["content"] = json.dumps(
                request_messages[0]["content"], indent=4
            )

            # create a folder for the code and logs, starts with code_gen_<timestamp>
            code_folder = iter_config["child"]
            os.makedirs(code_folder, exist_ok=True)

            response_message = await API_ENDPOINTS[args.api](
                request_messages, log_folder=code_folder
            )

            # for all files that named "prompt.txt" or ending with ".py" in the init_code_folder_name, read the content and write to code_folder_name
            for file in os.listdir(iter_config["parent"]):
                if file == "prompt.txt" or file.endswith(".py"):
                    with open(os.path.join(iter_config["parent"], file), "r") as f:
                        content = f.read()
                    with open(os.path.join(code_folder, file), "w") as f:
                        f.write(content)

            # match response.text with regex and extract the code blocks
            import re

            code_blocks = re.findall(
                r"```python\n(.*?)```", response_message, re.DOTALL
            )

            # print(code_blocks)
            # write the code blocks to files
            for code_block in code_blocks:
                filename = code_block.split("\n")[0].split("# filename: ")[1]
                print(f"writing file: {os.path.join(code_folder, filename)}")
                with open(os.path.join(code_folder, filename), "w") as f:
                    f.write(code_block)

            # extract test commands from response.text
            test_commands = re.findall(
                r".*?```bash\n(.*?)```.*?", response_message, re.DOTALL
            )
            # get the first test command if available
            if len(test_commands) > 0:
                TEST_COMMAND = test_commands[0].strip()
                # write test command to log file
                with open(os.path.join(code_folder, "test.command"), "w") as f:
                    f.write(TEST_COMMAND)
            cmd = f"bash -c 'cd {code_folder} && {TEST_COMMAND}'"
            print("--------------------------------")
            print(cmd)
            print("--------------------------------")
            result = await run_command_async(cmd)
            print("--------------------------------")
            print(cmd)
            print("--------------------------------")
            STDOUT_MESSAGE = result["stdout"]
            if STDOUT_MESSAGE != "":
                with open(os.path.join(code_folder, "code.stdout"), "w") as f:
                    f.write(STDOUT_MESSAGE)
            print(STDOUT_MESSAGE)
            print("--------------------------------")
            STDERR_MESSAGE = result["stderr"]
            if STDERR_MESSAGE != "":
                with open(os.path.join(code_folder, "code.stderr"), "w") as f:
                    f.write(STDERR_MESSAGE)
            print(STDERR_MESSAGE)
            print("--------------------------------")
            # get the exit code
            exit_code = result["returncode"]
            if exit_code != 0:
                print(f"Test failed with exit code {exit_code} [{code_folder}]")
                # write ".failed" to the code_folder_name
                with open(os.path.join(code_folder, ".failed"), "w") as f:
                    f.write(f"{exit_code}")
                print("--------------------------------")
                print(f"Fixing code... [{code_folder}]")
                print("--------------------------------")
            else:
                SUCCESS = True
                print(f"Test passed with exit code {exit_code} [{code_folder}]")
                # write ".success" to the code_folder_name
                with open(os.path.join(code_folder, ".success"), "w") as f:
                    f.write(f"{exit_code}")
                print("--------------------------------")
                break

        except Exception as e:
            exception_count += 1
            print(f"Exception [{code_folder}]: {e}")
            # print stacktrace
            print("--------------------------------")
            import traceback

            traceback.print_exc()
            # write stacktrace to log file
            with open(os.path.join(code_folder, "api_exception.txt"), "a") as f:
                f.write("--------------------------------\n")
                f.write(traceback.format_exc())
                f.write("--------------------------------\n")
            print("--------------------------------")
            if exception_count >= args.max_exceptions:
                print(f"Processor {id}: max exceptions reached [{code_folder}]")
                break
            else:
                await asyncio.sleep(3)
                continue

    print("--------------------------------")
    print(f"Processor {id} finished")
    print("--------------------------------")


async def main(args):

    queue = asyncio.Queue()
    folder_name = get_folder_name(args)

    await initialize_queue(args, folder_name, queue)

    tasks = []
    for i in range(args.parallel):
        task = asyncio.create_task(processor(args, queue, i))
        tasks.append(task)
        # sleep for 3 seconds
        await asyncio.sleep(3)

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--api", type=str, default="deepseek-fw")
    parser.add_argument("-d", "--dir", type=str, default="./func_matmul")
    parser.add_argument("-b", "--breadth", type=int, default=4)
    parser.add_argument("-t", "--depth", type=int, default=3)
    parser.add_argument("-p", "--parallel", type=int, default=2)
    parser.add_argument("-x", "--max-exceptions", type=int, default=3)
    args = parser.parse_args()

    if args.api not in API_ENDPOINTS.keys():
        print(f"Invalid API: {args.api} (available APIs: {API_ENDPOINTS.keys()})")
        exit(1)

    # check if directory exists
    if not os.path.exists(args.dir):
        print(f"Directory {args.dir} does not exist")
        exit(1)

    # check if prompt.txt exists
    if not os.path.exists(os.path.join(args.dir, "prompt.txt")):
        print(f"Prompt file {os.path.join(args.dir, 'prompt.txt')} does not exist")
        exit(1)

    # check if test.command exists
    if os.path.exists(os.path.join(args.dir, "test.command")):
        with open(os.path.join(args.dir, "test.command"), "r") as f:
            TEST_COMMAND = f.read()

    asyncio.run(main(args))
