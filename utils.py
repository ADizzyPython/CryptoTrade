import os
import sys
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
import os
os.environ['OPENAI_API_KEY'] = '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'
# ---
# This code is hardwired for OpenRouter. Set OPENAI_API_KEY to your OpenRouter key.
# Example model names: 'openai/gpt-4o', 'google/gemini-2.0-flash-exp:free', 'google/gemini-pro', etc.
# See https://openrouter.ai/docs for full model list and usage.
# ---

api_key = os.getenv('OPENAI_API_KEY')
api_base = 'https://openrouter.ai/api/v1'
client = OpenAI(api_key=api_key, base_url=api_base)

from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)

from typing import Optional, List
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

# Example: Model = Literal["openai/gpt-4o", "google/gemini-pro", ...]
Model = str

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_chat(prompt, model, seed, temperature=0.0, max_tokens=256, stop_strs=None, is_batched=False, debug=False):
    messages: List[ChatCompletionUserMessageParam] = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        seed=seed,
        temperature=temperature
        # max_tokens=max_tokens,  # Uncomment if you want to control output length
        # stop=stop_strs,         # Uncomment if you want to use stop sequences
    )
    if debug:
        print(response.system_fingerprint)
    return response.choices[0].message.content

if __name__ == "__main__":
    print(client.api_key[-4:])
    response = get_chat(
        "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair. Compose a poem that explains the concept of recursion in programming.",
        "openai/gpt-3.5-turbo", 6216, debug=True)
    print(response)
