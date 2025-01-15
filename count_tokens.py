import json
from pathlib import Path
import os

json_paths = list(Path("new_prompts").rglob("*_dump.json"))

total_count = 0
total_completion_tokens = 0
total_prompt_tokens = 0

for path in json_paths:
    di = json.load(open(path))
    ct = di['usage']['completion_tokens']
    pt = di['usage']['prompt_tokens']

    total_completion_tokens += ct
    total_prompt_tokens += pt

    total_count += 1

mean_completion_tokens = total_completion_tokens / total_count
mean_prompt_tokens = total_prompt_tokens / total_count