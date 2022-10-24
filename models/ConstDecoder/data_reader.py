
#This data reader are modified from the following two files:
#https://github.com/NetEase-GameAI/SARG/blob/master/run_train.py
#https://github.com/NetEase-GameAI/SARG/blob/master/data_utils.py

import json
import copy

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

TAGS = {
    "DELETE": 0,
    "KEEP": 1,
    "CHANGE": 2
}


def _compute_lcs(source, target):
    """Computes the Longest Common Subsequence (LCS).

  Description of the dynamic programming algorithm:
  https://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    source: List of source tokens.
    target: List of target tokens.

  Returns:
    List of tokens in the LCS.
  """
    table = _lcs_table(source, target)
    return _backtrack(table, source, target, len(source), len(target))


def _lcs_table(source, target):
    """Returns the Longest Common Subsequence dynamic programming table."""
    rows = len(source)
    cols = len(target)
    lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if source[i - 1] == target[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
    return lcs_table


def _backtrack(table, source, target, i, j):
    """Backtracks the Longest Common Subsequence table to reconstruct the LCS.

  Args:
    table: Precomputed LCS table.
    source: List of source tokens.
    target: List of target tokens.
    i: Current row index.
    j: Current column index.

  Returns:
    List of tokens corresponding to LCS.
  """
    if i == 0 or j == 0:
        return []
    if source[i - 1] == target[j - 1]:
        # Append the aligned token to output.
        return _backtrack(table, source, target, i - 1, j - 1) + [target[j - 1]]
    if table[i][j - 1] > table[i - 1][j]:
        return _backtrack(table, source, target, i, j - 1)
    else:
        return _backtrack(table, source, target, i - 1, j)


def insert_dummy(tokens, p='[unused%d]'):
    rlt = []
    cnt = 1
    for token in tokens:
        rlt.append(p % cnt)
        rlt.append(token)
        cnt += 1
    rlt.append(p % cnt)
    return rlt

def convert_tokens_to_string(tokenizer, tokens):
    return tokenizer.convert_tokens_to_string(tokens)


def _decode_valid_tags(source, tags, tokenizer):
    string = []
    for token, tag in zip(source, tags):
        if tag == 'DELETE':
            continue
        elif tag == 'KEEP':
            string.append(token)
        else:
            string.append(tag.split('|')[-1])
    return convert_tokens_to_string(tokenizer, string)


def convert_tags(source, target, tokenizer, debug=False):

    source = insert_dummy(tokenizer.tokenize(source))
    target = tokenizer.tokenize(target)

    # initialize tags
    tags = ['DELETE'] * len(source)

    kept_tokens = _compute_lcs(source, target) + ['[DUMMY]']

    target_idx = 0
    phrase = []

    for source_idx in range(len(source)):
        if source[source_idx] == kept_tokens[0]:
            tags[source_idx] = 'KEEP'
            while target_idx < len(target) and target[target_idx] != kept_tokens[0]:
                phrase.append(target[target_idx])
                target_idx += 1

            kept_tokens = kept_tokens[1:]

            if len(phrase) > 0:
                if debug:
                    tags[source_idx - 1] = 'CHANGE|' + convert_tokens_to_string(tokenizer, phrase)
                else:
                    tags[source_idx - 1] = 'CHANGE|' + '<|>'.join(phrase)
                phrase = []

            target_idx += 1

    if target_idx < len(target):
        if debug:
            tags[-1] = 'CHANGE|' + convert_tokens_to_string(tokenizer, target[target_idx:])
        else:
            tags[-1] = 'CHANGE|' + "<|>".join(target[target_idx:])

    if debug and _decode_valid_tags(source, tags, tokenizer) != convert_tokens_to_string(tokenizer, target):
        print(f"decoded: {_decode_valid_tags(source, tags, tokenizer)} "
              f"original: {convert_tokens_to_string(tokenizer, target)}")
    return tags, source


class SimpleTokenizer:
    def __init__(self, separator):
        self.separator = separator


    def tokenize(self, text):
        return text.strip().split(self.separator)


    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)


def further_convert_tags(src, tags, tokenizer):
    wordpiece_token_lst = []
    wordpiece_label_lst = []
    for token_item, label_item in zip(src, tags):
        tmp_token_lst = tokenizer.tokenize(token_item)
        if len(tmp_token_lst) == 1:
            wordpiece_token_lst.append(tmp_token_lst[0])
            if "CHANGE" in label_item:
                temp_label = label_item.replace("CHANGE|", "").replace("<|>", " ")
                label_item = 'CHANGE|' + "<|>".join(tokenizer.tokenize(temp_label))
            wordpiece_label_lst.append(label_item)
        else:
            len_wordpiece = len(tmp_token_lst)
            wordpiece_token_lst.extend(tmp_token_lst)
            tmp_label_lst = [label_item] * len_wordpiece
            wordpiece_label_lst.extend(tmp_label_lst)

    return wordpiece_label_lst, wordpiece_token_lst

def data_iter(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for sample in data.values():
            source = sample["ASR"]
            target = sample["RAW"]
            if not source:  # for some cases where source is none
                continue
            yield source, target


def get_examples(examples_path, tokenizer, max_src_len, max_add_len):
    simple_tokenizer = SimpleTokenizer(" ")
    examples = {
        "src_token": [],
        "src_mask": [],
        "src_pos": [],
        "target": [],
        "RAW": [],
        "ASR": []
    }


    for source, target in data_iter(examples_path):
        ASR, RAW = copy.deepcopy(source), copy.deepcopy(target)
        tags, src = convert_tags(source, target, simple_tokenizer, debug=False)
        tags, src = further_convert_tags(src, tags, tokenizer)

        src = tokenizer.convert_tokens_to_ids(src)[:max_src_len]
        tags = tags[:max_src_len]
        src = [tokenizer.cls_token_id] + src + [tokenizer.sep_token_id]
        tags = ["DELETE"] + tags + ["DELETE"]

        src_mask = [1] * len(src)
        src_pos = [x for x in range(len(src))]

        target = [[] for _ in range(len(src))]
        for idx in range(len(src)):
            if tags[idx].startswith("CHANGE|"):
                add = ["[CLS]"] + tags[idx][len("CHANGE|"):].split("<|>")
                add = tokenizer.convert_tokens_to_ids(add)[:max_add_len-1] + [102]
                target[idx].extend([TAGS["CHANGE"]] + add)
            else:
                target[idx].append(TAGS[tags[idx]])

        examples["src_token"].append(src)
        examples["src_mask"].append(src_mask)
        examples["src_pos"].append(src_pos)
        examples["target"].append(target)

        examples["ASR"].append(ASR)
        examples["RAW"].append(RAW)

    return examples


class ExampleDataset(Dataset):
    def __init__(self, dict_examples):
        self.examples = dict_examples

    def __len__(self):
        return len(list(self.examples.values())[0])

    def __getitem__(self, i):
        single_example = {}
        for k, v in self.examples.items():
            single_example.update({k: v[i]})
        return single_example


def collate_batch(batch_dict_examples):
    inputs = {}
    src_token = []
    src_mask = []
    src_pos = []
    target = []
    src_max_len = -1
    add_max_len = -1

    for x in batch_dict_examples:
        src_max_len = max(len(x["src_token"]), src_max_len)
        for t in x["target"]:
            add_max_len = max(add_max_len, len(t))

    for dict_example in batch_dict_examples:
        src_token.append(dict_example["src_token"] + [0] * (src_max_len - len(dict_example["src_token"])))
        src_mask.append(dict_example["src_mask"] + [0] * (src_max_len - len(dict_example["src_token"])))
        src_pos.append(dict_example["src_pos"] + [p for p in range(len(dict_example["src_pos"]), src_max_len)])

        _tgt = []
        for t in dict_example["target"]:
            _tgt.append(t + [0] * (add_max_len - len(t)))

        target.append(_tgt + [[0] * add_max_len] * (src_max_len - len(dict_example["src_token"])))

    inputs["src_token"] = torch.tensor(src_token, dtype=torch.long)
    inputs["src_mask"] = torch.tensor(src_mask, dtype=torch.long)
    inputs["src_pos"] = torch.tensor(src_pos, dtype=torch.long)
    inputs["target"] = torch.tensor(target, dtype=torch.long)

    inputs["ASRS"] = [item["ASR"] for item in batch_dict_examples]
    inputs["RAWS"] = [item["RAW"] for item in batch_dict_examples]

    return inputs
