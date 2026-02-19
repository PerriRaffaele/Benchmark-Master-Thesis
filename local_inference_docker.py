import re
from litellm import completion
from datasets import load_dataset
import subprocess
import tempfile
import os

def parse_code_block(string: str) -> str:
    """
    Parses a code block (if present) from a string.

    Args:
        string: The input string which may contain a markdown code block.

    Returns:
        The content of the first code block found, or the full string if no valid
        markdown code block (```...```) is present.
    """
    code_pattern = r"```[^\n]*\n(.*?)\n```"
    match = re.search(code_pattern, string, re.DOTALL)
    if match:
        return match.group(1)
    else:
        print("No code block found; returning full string.")
        return string


def generate(prompt: str, **args):
    messages = [
        {"role": "system", "content": f"""You are an AI that only responds with Python code, NOT ENGLISH. You will be given a function signature and its docstring by the user. Write your full implementation. You always return the signature and anything that came before it in the input prompt (such as the docstring, libraries, imports, and so on) along with the full implementation of the function. Write the output in a markdown code block. For example:\n \n<your code here>\n"""},
        {"role": "user", "content": prompt}
    ]

    response = completion(
        **args,
        messages=messages,
        caching=False,
        cache={"no-cache": True, "no-store": True}
    )

    return parse_code_block(response.choices[0].message.content)


if __name__ == '__main__':
    # model = 'hosted_vllm/unsloth/Qwen2.5-Coder-1.5B-Instruct'
    model = 'ollama/llama2'
    completion_kwargs = {
        "model": model,
        "max_tokens": 1024,
        "temperature": 0.2,
        # 'api_base': 'http://localhost:8000/v1',
        'api_base': "http://localhost:11434",
    }

    print("Checking Docker image (Downloading if missing)...")
    try:
        subprocess.run(["docker", "pull", "python:3.10-slim"], check=True, capture_output=True)
        print("Docker image is ready!\n")
    except subprocess.CalledProcessError as e:
        print(f"Failed to pull Docker image. Is Docker Desktop running? Error: {e}")
        exit(1)

    ds = load_dataset("openai/openai_humaneval")
    correct = 0
    total = 0
    for task in ds['test']:
        total += 1
        print(f"Task ID: {task['task_id']}")
        print(f"Prompt:\n{task['prompt']}")
        output = generate(task['prompt'], **completion_kwargs)
        print(f"Output:\n{output}\n")
        if "METADATA" in task['test']:
            test_code = re.search(r"(def check\(candidate\):.*?)(?= def |\Z)", task['test'], re.DOTALL)
            if test_code:
                test_code = test_code.group(1)
            else:
                print("No check(candidate) function found in test code; using full test code.")
        else:
            test_code = task['test']
        code = f"{task['prompt']}\n{output}\n{test_code}\ncheck({task['entry_point']})\n"
        print(f"Tests to run:\n{test_code}\n")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_script:
            temp_script.write(code)
            temp_script_path = os.path.abspath(temp_script.name)

        try:
            result = subprocess.run(
                [
                    "docker", "run",
                    "--rm", "-i",
                    "--net", "none",
                    "--memory", "256m",
                    "-v", f"{temp_script_path}:/app/script.py:ro", # :ro for read-only
                    "python:3.10-slim",
                    "python", "/app/script.py"
                ],
                input=code,
                text=True,  # String input
                capture_output=True,
                timeout=10  # Kills the container if it takes longer than 10 seconds
            )

            # 5. Check the status code
            if result.returncode == 0:
                print("  Test passed!")
                correct += 1
            else:
                print(f"  Test failed! (Exit Code {result.returncode})")
                error_lines = result.stderr.strip().split('\n')
                print(f"  Error: {error_lines[-1] if error_lines else 'Unknown error'}")

        except subprocess.TimeoutExpired:
            print("  Code took too long, likely an infinite loop")
        except Exception as e:
            print(f"  Error: ({e})")
        finally:
            if os.path.exists(temp_script_path):
                os.remove(temp_script_path)

    accuracy = correct / total
    print(f"Final Accuracy: {accuracy:.2%}")
