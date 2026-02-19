import re
from litellm import completion
from datasets import load_dataset


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
        "max_tokens": 256,
        "temperature": 0.2,
        # 'api_base': 'http://localhost:8000/v1',
        'api_base': "http://localhost:11434",
    }

    # print(generate("Write a Python function that adds two numbers.", **completion_kwargs))
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
        execution_env = {}
        try:
            exec(task['prompt'], execution_env)
            exec(code, execution_env)
            correct += 1
        except Exception as e:
            print(f"Error executing code for task {task['task_id']}: {e}")
    accuracy = correct / total
    print(f"Final Accuracy: {accuracy:.2%}")
