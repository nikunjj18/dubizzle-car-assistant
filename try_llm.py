import time

import litellm
from dotenv import load_dotenv

load_dotenv()
litellm.drop_params = True


def run(label, **extra):
    start = time.time()
    try:
        resp = litellm.completion(
            model="gemini/gemini-3.8-flash",
            messages=[{"role": "user", "content": "Reply with exactly: key works"}],
            timeout=90,
            **extra,
        )
        print(f"{label}: OK in {time.time() - start:.1f}s -> {resp.choices[0].message.content!r}")
    except Exception as e:
        print(f"{label}: {type(e).__name__} after {time.time() - start:.1f}s -> {str(e)[:300]}")


run("default thinking")
run("low reasoning", reasoning_effort="low")