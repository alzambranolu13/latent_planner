from datetime import datetime
import json
import os
import time
from pathlib import Path
import weblinx as wl
from pathlib import Path
import time
from tqdm import tqdm
from os import listdir
from os.path import isfile, join, isdir


def load_json_no_cache(basedir, name):
    if not os.path.exists(f"{basedir}/{name}.json"):
        return None
    try:
        with open(f"{basedir}/{name}.json", "r") as f:
            j = json.load(f)
    except:
        return None
    
    return j


def load_recording(basedir):
    # Before loading replay, we need a dropdown that allows us to select replay.json or replay_orig.json
    # Find all files in basedir starting with "replay" and ending with ".json"
    replay_files = sorted(
        [
            f
            for f in os.listdir(basedir)
            if f.startswith("replay") and f.endswith(".json")
        ]
    )
    replay_file = 'replay.json'
    replay_file = replay_file.replace(".json", "")

    metadata = load_json_no_cache(basedir, "metadata")


    # Read in the JSON data
    replay_dict = load_json_no_cache(basedir, replay_file)


    data = replay_dict["data"]
    return data,metadata

def format_chat_message(d):
    if d["speaker"] == "instructor":
        return "🧑 " + d["utterance"]
    else:
        return "🤖 " + d["utterance"]
    
def shorten(s):
    # shorten to 100 characters
    if len(s) > 100:
        s = s[:100] + "..."

    return s
    
def parse_arguments(action):
    s = []
    event_type = action["intent"]
    args = action["arguments"]

    if event_type == "textInput":
        txt = args["text"]

        txt = txt.strip()

        # escape markdown characters
        txt = txt.replace("_", "\\_")
        txt = txt.replace("*", "\\*")
        txt = txt.replace("`", "\\`")
        txt = txt.replace("$", "\\$")

        txt = shorten(txt)

        s.append(f'"{txt}"')
    elif event_type == "change":
        s.append(f'{args["value"]}')
    elif event_type == "load":
        url = args["properties"].get("url") or args.get("url")
        short_url = shorten(url)
        s.append(f'"[{short_url}]({url})"')

        if args["properties"].get("transitionType"):
            s.append(f'*{args["properties"]["transitionType"]}*')
            s.append(f'*{" ".join(args["properties"]["transitionQualifiers"])}*')
    elif event_type == "scroll":
        s.append(f'{args["scrollX"]}, {args["scrollY"]}')
    elif event_type == "say":
        s.append(f'"{args["text"]}"')
    elif event_type == "copy":
        selected = shorten(args["selected"])
        s.append(f'"{selected}"')
    elif event_type == "paste":
        pasted = shorten(args["pasted"])
        s.append(f'"{pasted}"')
    elif event_type == "tabcreate":
        s.append(f'{args["properties"]["tabId"]}')
    elif event_type == "tabremove":
        s.append(f'{args["properties"]["tabId"]}')
    elif event_type == "tabswitch":
        s.append(
            f'{args["properties"]["tabIdOrigin"]} -> {args["properties"]["tabId"]}'
        )

    if args.get("element"):
        
        if event_type == 'click':
            x = round(args['metadata']['mouseX'], 1)
            y = round(args['metadata']['mouseY'], 1)
            uid = args.get('element', {}).get('attributes', {}).get("data-webtasks-id")
            s.append(f"*x =* {x}, *y =* {y}, *uid =* {uid}")
        else:
            top = round(args["element"]["bbox"]["top"], 1)
            left = round(args["element"]["bbox"]["left"], 1)
            right = round(args["element"]["bbox"]["right"], 1)
            bottom = round(args["element"]["bbox"]["bottom"], 1)
            
            s.append(f"*top =* {top}, *left =* {left}, *right =* {right}, *bottom =* {bottom}")

    return ", ".join(s)


def create_prompt(mainPrompt,mainResponse,prompt_dir):
    onlyfiles = [f for f in listdir(prompt_dir) if isfile(join(prompt_dir, f))]
    len_mes= int((len(onlyfiles)-1)/2)
    messages=[{"role": "user", "content": mainPrompt}, {"role": "assistant", "content": mainResponse}]
    for i in range(len_mes):
        question = open(f'{prompt_dir}/question{i}.txt', "r").read()    
        answer = open(f'{prompt_dir}/answer{i}.txt', "r").read()
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

    return messages
    


# %%
import json
import re
import openai
from openai import OpenAI


class ParseError(Exception):
    pass


def extract_html_tags(text, keys):
    """Extract the content within HTML tags for a list of keys.

    All text and keys will be converted to lowercase before matching.

    Args:
        text (str): The input string containing the HTML tags.
        keys (list[str]): The HTML tags to extract the content from.

    Returns:
        dict: A dictionary mapping each key to a list of subset in `text` that match the key.
    """
    content_dict = {}
    # text = text.lower()
    # keys = set([k.lower() for k in keys])
    for key in keys:
        pattern = f"<{key}>(.*?)</{key}>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict

def parse_html_tags(text, keys=(), optional_keys=(), merge_multiple=False):
    """Satisfy the parse api, extracts 1 match per key and validates that all keys are present

    Args:
        text (str): The input string containing the HTML tags.
        keys (list[str]): The HTML tags to extract the content from.
        optional_keys (list[str]): The HTML tags to extract the content from, but are optional.
        merge_multiple (bool): Whether to merge multiple instances of the same key.

    Returns:
        dict: A dictionary mapping each key to a subset of `text` that match the key.
        bool: Whether the parsing was successful.
        str: A message to be displayed to the agent if the parsing was not successful.

    """
    all_keys = tuple(keys) + tuple(optional_keys)
    content_dict = extract_html_tags(text, all_keys)
    retry_messages = []

    for key in all_keys:
        if not key in content_dict:
            if not key in optional_keys:
                retry_messages.append(f"Missing the key <{key}> in the answer.")
        else:
            val = content_dict[key]
            content_dict[key] = val[0]
            if len(val) > 1:
                if not merge_multiple:
                    retry_messages.append(
                        f"Found multiple instances of the key {key}. You should have only one of them."
                    )
                else:
                    # merge the multiple instances
                    content_dict[key] = "\n".join(val)

    valid = len(retry_messages) == 0
    retry_message = "\n".join(retry_messages)
    return content_dict, valid, retry_message

def parse_html_tags_raise(text, keys=(), optional_keys=(), merge_multiple=False):
    """A version of parse_html_tags that raises an exception if the parsing is not successful."""
    content_dict, valid, retry_message = parse_html_tags(
        text, keys, optional_keys, merge_multiple=merge_multiple
    )
    if not valid:
        raise ParseError(retry_message)
    return content_dict

def adapt_text(text, n_retry):
  client = OpenAI(api_key=os.environ.get("OPENAI"))
  tries= 0
  while tries < n_retry:
    try:
      prompt_dir= 'prompts/general'
      mainPrompt= open(f'{prompt_dir}/mainPrompt.txt').read()
      mainResponse= open(f'{prompt_dir}/mainResponse.txt').read()
      messages = create_prompt(mainPrompt,mainResponse,prompt_dir)
      messages.append({"role": "user", "content": text})
      response = client.chat.completions.create(
                    model="gpt-4o-2024-08-06",
                    temperature=0,
                    messages = messages,
                    seed = 42
                )
      reply_content = response.choices[0].message.content
      model_dump = response.model_dump_json()
      ans_dict = parse_html_tags_raise(reply_content,keys=('goal','steps') )
      return ans_dict,model_dump
    except ParseError as parsing_error:
            tries += 1

  raise ParseError(f"Could not parse a valid value after {n_retry} retries.")
        


def main():
    wl_dir = Path("./wl_data")
    base_dir = wl_dir / "demonstrations"
    split_path = "splits.json"
    prompt_dir= f'4o_extracted_data'
    splits=None
    with open(split_path) as f:
        splits= json.load(f)
    demo_names = splits['train']
    demos = [wl.Demonstration(name, base_dir=base_dir) for name in demo_names]

    for demo in tqdm(demos):
        replay = wl.Replay.from_demonstration(demo)
        turns = replay.filter_by_type("chat")
        if len(turns) == 0:
            continue
        chat= ""
        for turn in turns:    
            chat+=(format_chat_message(turn))+'\n'
        #print(stringBuilder)
        ans_dict, model_dump= adapt_text(chat,3)
        with open(f'{prompt_dir}/{demo.name}_goal.txt', 'w+') as f:
            f.write(ans_dict['goal'])
        with open(f'{prompt_dir}/{demo.name}_steps.txt', 'w+') as f:
            f.write(ans_dict['steps'])
        with open(f'{prompt_dir}/{demo.name}_dump.json', 'w+') as f:
            f.write(model_dump)
        
        #print('--------------------------------------------------------')
            #time.sleep(3)


if __name__ == "__main__":
    main()
