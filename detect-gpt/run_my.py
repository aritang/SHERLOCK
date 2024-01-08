import matplotlib.pyplot as plt
import numpy as np
import datasets
import transformers
import re
import torch
import torch.nn.functional as F
import tqdm
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import argparse
import datetime
import os
import json
import functools
# import custom_datasets
from multiprocessing.pool import ThreadPool
import time
from transformers import T5Model, T5Tokenizer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("using device", DEVICE)
CHN = False
CODE = True
N = 216
M = 216
preferred_max_length = 1024
RESULT_FOLDER = "T5LARGE_OPT150_VENTI"
MAX_MASK = 1000
IF_GPT = True
CHUNK_SIZE = 12

def load_base_model(model_name = "facebook/opt-150m", cache_dir="model-cache"):
    print('MOVING BASE MODEL TO G/CPU...', end='', flush=True)
    start = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model.to(DEVICE), tokenizer


def load_mask_model(model_name = "t5-large", cache_dir="model-cache"):
    # print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    # start = time.time()
    
    # model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    # print(f'DONE ({time.time() - start:.2f}s)')
    # return model.to(DEVICE), tokenizer

    print('LOADING MASK MODEL...', end='', flush=True)
    start = time.time()

    # Load T5 model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, model_max_length=preferred_max_length)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print(f'DONE ({time.time() - start:.2f}s)')
    return model.to(DEVICE), tokenizer



"""
Masking: tokenize_and_mask masks certain words in the input text, replacing them with special tokens (like <extra_id_NUM>).

Filling Masks: replace_masks replaces each masked span with a sample from the T5 model. It uses the mask_model to generate text that fills in the masked parts.

Extracting Fills: extract_fills extracts the filled portions from the generated text.

Applying Fills: apply_extracted_fills applies the extracted fills back into the original masked texts.

Perturbing Texts: perturb_texts_ and perturb_texts use the above functions to create perturbed versions of the input texts. It handles retries if the model fails to generate the correct number of fills and deals with chunking for large datasets.
"""

# def parse_code(code_str):
#     tokens = []
#     current_token = ''
#     current_space = ''

#     for char in code_str:
#         if char == ' ':
#             if current_token:
#                 tokens.append(current_token)
#                 current_token = ''
#             current_space += char
#         elif char in ['\n', '\t']:
#             if current_token:
#                 tokens.append(current_token)
#                 current_token = ''
#             if current_space:
#                 tokens.append(current_space)
#                 current_space = ''
#             tokens.append(char)
#         else:
#             if current_space:
#                 tokens.append(current_space)
#                 current_space = ''
#             current_token += char

#     # Add any remaining token or space
#     if current_token:
#         tokens.append(current_token)
#     elif current_space:
#         tokens.append(current_space)

#     return tokens

"""
def parse_code(code_str):
    # Tokenize while keeping spaces, newlines, and tabs as separate tokens
    tokens = []
    current_token = ''
    for char in code_str:
        if char in [' ', '\n', '\t']:
            if current_token:
                tokens.append(current_token)
                current_token = ''
            tokens.append(char)
        else:
            current_token += char
    if current_token:
        tokens.append(current_token)
    return tokens
"""
token_pattern = r"""
    (?P<WHITESPACE>\s+)|                                # Match whitespaces
    (?P<SPECIAL>[\#\n])|                               # Match special characters like # and newlines
    (?P<STRING_LITERAL>\"(?:\\.|[^\"\\])*\"|\'.*?\')|  # Match string literals
    (?P<COMMENT>//.*?$|/\*.*?\*/)|                     # Match comments
    (?P<OTHER>\S+)                                     # Match other tokens (non-space)
"""
# No-white-space ver.
# def parse_code(code):
#     # Regular expression for C++ tokens
#     # Compile the regular expression with VERBOSE flag to allow whitespace and comments
#     token_regex = re.compile(token_pattern, re.MULTILINE | re.DOTALL | re.VERBOSE)

#     # Tokenize
#     tokens = [match.group() for match in token_regex.finditer(code)]
#     return tokens

def parse_code(code):
    # Regular expression for C++ tokens
    # Compile the regular expression with VERBOSE flag to allow whitespace and comments
    token_regex = re.compile(token_pattern, re.MULTILINE | re.DOTALL | re.VERBOSE)

    # Tokenize
    tokens = [match.group() for match in token_regex.finditer(code)]
    return tokens

"""
def tokenize_and_mask(text, span_length, pct, ceil_pct=False, buffer_size = 1):
    if CHN:
        tokens = list(text)
    elif CODE:
        tokens = parse_code(text)
    else:
        tokens = text.split(' ')

    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    if CHN or CODE:
        text = "".join(tokens)
    else:
        text = ' '.join(tokens)
    return text
"""

def tokenize_and_mask(text, span_length, pct, ceil_pct=False, buffer_size = 2, max_mask = 20):
    if CHN:
        tokens = list(text)
    elif CODE:
        tokens = parse_code(text)
    else:
        tokens = text.split(' ')

    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans and n_masks < max_mask:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    if CHN or CODE:
        text = "".join(tokens)
    else:
        text = ' '.join(tokens)
    return text

def count_masks(texts):
    if CHN or CODE:
        return [text.count("<extra_id_") for text in texts]
    else:
        return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]
    

# replace each masked span with a sample from T5 mask_model
def replace_masks(texts, mask_model, mask_tokenizer, mask_top_p = 1.0):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = mask_model.generate(**tokens, do_sample=True, max_length=preferred_max_length, top_p=mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)

pattern = re.compile(r"<extra_id_\d+>")
def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills

# Regular expression pattern to match the <extra_id_NUM> tokens
pattern_ = re.compile(r"<extra_id_\d+>")

def tokenize_text(text):
    tokens = []
    start = 0
    for match in pattern.finditer(text):
        # Add characters from the last match up to this one
        tokens.extend(list(text[start:match.start()]))
        # Add the <extra_id_NUM> token
        tokens.append(match.group())
        # Update the start position for the next iteration
        start = match.end()
    # Add any remaining characters after the last match
    tokens.extend(list(text[start:]))
    return tokens

def apply_extracted_fills(masked_texts, extracted_fills):
    # if CHN:
    #     tokens = [tokenize_text(text) for text in masked_texts]
    # elif CODE:
    #     tokens = [parse_code(text) for text in masked_texts]
    # else:
    #     # split masked text into tokens, only splitting on spaces (not newlines)
    if not (CHN or CODE):
        tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)
    texts = []

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(masked_texts, extracted_fills, n_expected)):
        for fill_idx in range(len(fills)):
            text = text.replace(f"<extra_id_{fill_idx}>", fills[fill_idx])
        if len(fills) < n:
            print(f"warning!!! in extracted_fills, len(fills)({len(fills)}) < n({n}), just adding spaces")
            for fill_idx in range(len(fills), n):
                text = text.replace(f"<extra_id_{fill_idx}>", " ")
        texts.append(text)
    # join tokens back into text
    if not (CHN or CODE):
        texts = [" ".join(x) for x in tokens]

    return texts

"""
def perturb_texts_(texts, span_length, pct, mask_model, mask_tokenizer, ceil_pct=False):
    # print("ori texts", texts)
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct, max_mask=MAX_MASK) for x in texts]
    # print(f"masked_texts f{masked_texts}")
    raw_fills = replace_masks(masked_texts, mask_model=mask_model, mask_tokenizer=mask_tokenizer)
    # print("raw_fills = replace_masks(...)", raw_fills)
    extracted_fills = extract_fills(raw_fills)
    # print("extracted_fills = extract_fills(...)\n", extracted_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
    # print(f"*raw_fills\n {raw_fills[0]}")
    # print(f"*extracted_fills sample\n {extracted_fills[0]}")
    # print(f"*masked_texts sample\n {masked_texts[0]}")
    # print(f"*perturbed_texts sample\n {perturbed_texts[0]}")
    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct, max_mask=MAX_MASK) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(masked_texts, mask_model=mask_model, mask_tokenizer=mask_tokenizer)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1
    return perturbed_texts
"""


def perturb_texts_(texts, span_length, pct, mask_model, mask_tokenizer, ceil_pct=False):
    start_time = time.time()
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct, max_mask=MAX_MASK) for x in texts]
    tokenize_and_mask_time = time.time() - start_time
    # print(f"Time for tokenize_and_mask: {tokenize_and_mask_time:.2f}s")

    start_time = time.time()
    raw_fills = replace_masks(masked_texts, mask_model=mask_model, mask_tokenizer=mask_tokenizer)
    replace_masks_time = time.time() - start_time
    print(f"Time for replace_masks: {replace_masks_time:.2f}s")

    start_time = time.time()
    extracted_fills = extract_fills(raw_fills)
    extract_fills_time = time.time() - start_time
    # print(f"Time for extract_fills: {extract_fills_time:.2f}s")

    start_time = time.time()
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
    apply_extracted_fills_time = time.time() - start_time
    # print(f"Time for apply_extracted_fills: {apply_extracted_fills_time:.2f}s")

    total_time = tokenize_and_mask_time + replace_masks_time + extract_fills_time + apply_extracted_fills_time
    # print(f"Total time before retries: {total_time:.2f}s")
    """
        attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')

        start_time = time.time()
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct, max_mask=MAX_MASK) for idx, x in enumerate(texts) if idx in idxs]
        retry_tokenize_and_mask_time = time.time() - start_time
        print(f"Retry {attempts}, Time for tokenize_and_mask: {retry_tokenize_and_mask_time:.2f}s")

        start_time = time.time()
        raw_fills = replace_masks(masked_texts, mask_model=mask_model, mask_tokenizer=mask_tokenizer)
        retry_replace_masks_time = time.time() - start_time
        print(f"Retry {attempts}, Time for replace_masks: {retry_replace_masks_time:.2f}s")

        start_time = time.time()
        extracted_fills = extract_fills(raw_fills)
        retry_extract_fills_time = time.time() - start_time
        print(f"Retry {attempts}, Time for extract_fills: {retry_extract_fills_time:.2f}s")

        start_time = time.time()
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        retry_apply_extracted_fills_time = time.time() - start_time
        print(f"Retry {attempts}, Time for apply_extracted_fills: {retry_apply_extracted_fills_time:.2f}s")

        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1
    """
    return perturbed_texts


"""
def perturb_texts(texts, span_length, pct, mask_model, mask_tokenizer, ceil_pct=False, chunk_size = 20):
    # # don't know what this is, muted
    # if '11b' in mask_filling_model_name:
        # chunk_size //= 2

    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct, mask_model=mask_model, mask_tokenizer=mask_tokenizer))
    return outputs
"""

# def perturb_texts(texts, span_length, pct, mask_model, mask_tokenizer, N=100, ceil_pct=False, chunk_size=20):
#     outputs = []
#     for text in tqdm.tqdm(texts, desc="Processing texts"):
#         perturbed_versions = []
#         for _ in range(N):
#             perturbed_texts = perturb_texts_([text], span_length, pct, ceil_pct=ceil_pct, mask_model=mask_model, mask_tokenizer=mask_tokenizer)
#             perturbed_versions.extend(perturbed_texts)
#         outputs.append(perturbed_versions)
#     return outputs


def perturb_texts(texts, span_length, pct, mask_model, mask_tokenizer, N=100, ceil_pct=False, chunk_size=12):
    outputs = []
    for _ in tqdm.tqdm(range(N), desc="Processing iterations"):
        outputs_ = []
        for i in range(0, len(texts), chunk_size):
            batch_texts = texts[i:min(len(texts), i + chunk_size)]
            perturbed_batch = perturb_texts_(batch_texts, span_length, pct, ceil_pct=ceil_pct, mask_model=mask_model, mask_tokenizer=mask_tokenizer)
            outputs_.extend(perturbed_batch)
        outputs.append(outputs_)
    return outputs


def drop_last_word(text):
    if CHN or CODE:
        return ''.join(text.split(' ')[:-1])
    else:
        return ' '.join(text.split(' ')[:-1])

def get_ll_from_model(text, model, tokenizer):
    if not isinstance(text, str):
        print(f"Warning: Expected text of type `str`, but got type `{type(text)}`. Returning None.")
        if isinstance(text, list):
            if not isinstance(text[0], str):
                return None
    with torch.no_grad():
        tokenized = tokenizer(text, return_tensors="pt").to(DEVICE)
        labels = tokenized.input_ids
        return -model(**tokenized, labels=labels).loss.item()

"""
def get_lls(text_list, model, tokenizer):
    return [get_ll_from_model(text, model, tokenizer) for text in text_list]
"""

def get_lls(text_list, model, tokenizer):
    results = []
    for item in text_list:
        if isinstance(item, str):
            # Process the string
            results.append(get_ll_from_model(item, model, tokenizer))
        elif isinstance(item, list):
            # Recursively process the list
            results.append(get_lls(item, model, tokenizer))
        else:
            # Handle non-string, non-list items (if needed)
            results.append(None)
    return results

base_model, base_tokenizer = load_base_model()
mask_model, mask_tokenizer = load_mask_model()

def load_processed_code_from_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def save_list_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
    
# code = load_processed_code_from_file("human_code.json")
# # code = load_processed_code_from_file("GPT_code.json")



# text_list = [code_[0] for code_ in code]
# label_list = [code_[1] for code_ in code]

# perturbed_texts = perturb_texts(text_list, span_length=1, pct=0.2, N=100, mask_model=mask_model, mask_tokenizer=mask_tokenizer)

# text_list = text_list + perturbed_texts
# save_list_to_json(text_list, os.path.join(RESULT_FOLDER, 'human_text_list.json'))

# # print("likelyhood result: ", f"{get_lls(text_list, model=base_model, tokenizer=base_tokenizer)}")

# result = get_lls(text_list, model=base_model, tokenizer=base_tokenizer)
# save_list_to_json(result, os.path.join(RESULT_FOLDER, "human_result.json"))
"""
code = load_processed_code_from_file("GPT_code.json")

text_list = [code_[0] for code_ in code]
label_list = [code_[1] for code_ in code]
"""
if IF_GPT:
    code = load_processed_code_from_file("llama_code_24.json")
    text_list = []
    for m in range(M):
        text_list.append(code[m])
else:
    data = load_processed_code_from_file("train_23.json")
    text_list = []
    for m in range(M):
        text_list.append(data[m]['tokens'])
# text_list =  ["cpp\n// Problem Name: Cheater's Nim\n\n// Input:\nint N;\nint a_1;\nint a_2;\n...\nint a_N;\n\n<extra_id_0> Output:\nint<extra_id_1>cheater_stones;\n<extra_id_2>// Logic:\nint main() {\n    cin >> N;\n<extra_id_3>  <extra_id_4>cin >> a_1;\n    cin >> a_2;\n    ...\n <extra_id_5>  cin >> <extra_id_6>\n\n    // Initialize variables\n    int<extra_id_7>cheater_stones = 0;\n    int remaining_stones = 0;\n\n  <extra_id_8> // Loop through all piles\n<extra_id_9>   for (int i = 1; i <= N;<extra_id_10>i++) {\n<extra_id_11>      <extra_id_12>// If there are no stones left <extra_id_13> the pile, skip it\n        if (a_i <extra_id_14> 0) continue;\n<extra_id_15>     <extra_id_16> <extra_id_17>// Compute<extra_id_18>the maximum number <extra_id_19> stones the<extra_id_20>cheater can take from this pile\n        int<extra_id_21>max_stones = min(a_i, remaining_stones);\n\n        // Update the number<extra_id_22>of remaining stones\n    <extra_id_23>   remaining_stones -= max_stones;\n\n       <extra_id_24>// Update the number of stones the cheater has eaten\n        cheater_stones +=<extra_id_25>max_stones;\n\n        //<extra_id_26>Print the number of stones the cheater <extra_id_27> eaten<extra_id_28>so far\n      <extra_id_29> cout << cheater_stones <extra_id_30> endl;\n\n        // If there are no more stones left, break\n        if (remaining_stones == 0) break;\n    }\n\n  <extra_id_31> // If the cheater has eaten all the stones, print -1\n    if (cheater_stones == 0) cout << -1<extra_id_32><< <extra_id_33>\n\n    return 0;\n}"] 

perturbed_texts = perturb_texts(text_list, span_length=1, pct=0.2, N=N, mask_model=mask_model, mask_tokenizer=mask_tokenizer, chunk_size=CHUNK_SIZE)

text_list = text_list + perturbed_texts

os.makedirs(RESULT_FOLDER, exist_ok=True)
if IF_GPT:
    save_list_to_json(text_list, os.path.join(RESULT_FOLDER, 'GPT_text_list_24.json'))
else:
    save_list_to_json(text_list, os.path.join(RESULT_FOLDER, 'human_text_list_24.json'))
# print("likelyhood result: ", f"{get_lls(text_list, model=base_model, tokenizer=base_tokenizer)}")

result = get_lls(text_list, model=base_model, tokenizer=base_tokenizer)
print(f"run finished, saving to {RESULT_FOLDER}")

if IF_GPT:
    save_list_to_json(result, os.path.join(RESULT_FOLDER, "GPT_result_24.json"))
else:
    save_list_to_json(result, os.path.join(RESULT_FOLDER, "human_result_24.json"))
# print("testing ...ll func result: ", f"{get_ll_from_model(text_list, model=base_model, tokenizer=base_tokenizer)}")
