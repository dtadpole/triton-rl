import asyncio
import sys
import time
import os

async def run_command(cmd, name=""):
    """Run a shell command asynchronously and capture its output."""
    print(f"Starting {name}...")
    start_time = time.time()

    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    end_time = time.time()
    duration = end_time - start_time

    return {
        'name': name,
        'stdout': stdout.decode('utf-8'),
        'stderr': stderr.decode('utf-8'),
        'returncode': process.returncode,
        'duration': duration
    }

async def main():
    # List of commands to run with descriptive names
    commands = [
        # System information commands
        ('uname -a', 'System Info'),
        ('lscpu | head -n 5', 'CPU Info'),
        ('free -h', 'Memory Info'),
        ('df -h /', 'Disk Space'),

        # Python environment commands
        ('python --version', 'Python Version'),
        ('pip list | grep torch', 'PyTorch Version'),
        ('pip list | grep triton', 'Triton Version'),

        # GPU commands (if available)
        ('nvidia-smi', 'GPU Info'),

        # File operations
        ('ls -la /home/centos/triton-rl', 'List Triton-RL Dir'),
        ('find /home/centos/triton-rl -name "*.py" | wc -l', 'Count Python Files'),

        # Network commands
        ('curl -s ifconfig.me', 'Public IP'),
        ('ping -c 1 google.com', 'Network Test'),

        # Simulated long-running tasks
        ('sleep 3 && echo "Long task 1 done"', 'Long Task 1'),
        ('sleep 2 && echo "Long task 2 done"', 'Long Task 2'),
    ]

    print("Starting all commands...")
    start_time = time.time()

    # Run commands concurrently
    tasks = [run_command(cmd, name) for cmd, name in commands]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Print results
    print("\nResults:")
    print("-" * 50)

    for result in results:
        if isinstance(result, Exception):
            print(f"\nCommand failed with exception: {result}")
            continue

        print(f"\n{result['name']} (took {result['duration']:.2f}s):")
        print(f"Return code: {result['returncode']}")
        if result['stdout']:
            print("Output:")
            print(result['stdout'].strip())
        if result['stderr']:
            print("Errors:")
            print(result['stderr'].strip())
        print("-" * 50)

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print(f"Number of commands executed: {len(commands)}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
