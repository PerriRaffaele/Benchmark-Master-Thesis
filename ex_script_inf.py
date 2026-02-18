import re

from litellm import completion

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
        {"role": "system", "content": "You are an expert software engineer."},
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

    model = 'hosted_vllm/unsloth/Qwen2.5-Coder-1.5B-Instruct'

    completion_kwargs = {
        "model": model,
        "max_tokens": 256,
        "temperature": 0.2,
        'api_base': 'http://localhost:8000/v1',
    }

    print(generate("Write a Python function that adds two numbers.", **completion_kwargs))