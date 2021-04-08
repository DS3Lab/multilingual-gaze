import json
import os

import torch
from transformers import BertTokenizer, XLMTokenizer


def mask_mse_loss(b_output, b_target, target_pad, d_out):
    """
    Masks the pad tokens of by setting the corresponding output and target tokens equal.
    """
    active_mask = b_target.view(-1, d_out) == target_pad
    active_outputs = b_output.view(-1, d_out)
    active_targets = torch.where(active_mask, active_outputs, b_target.view(-1, d_out))

    return active_outputs, active_targets


def mask_ce_loss(b_output, b_target, target_pad, d_out, criterion):
    """
    Masks the pad tokens of by setting the corresponding target tokens
    to an index that is ignored by the CrossEntropyLoss.
    """
    active_mask = b_target.view(-1) == target_pad
    active_outputs = b_output.view(-1, d_out)
    active_targets = torch.where(active_mask, torch.tensor(criterion.ignore_index).type_as(b_target), b_target.view(-1))

    return active_outputs, active_targets


def split_gaze_batch(b_gaze_input):
    """
    Returns the indices in the range of b_gaze_input where a valid gaze vector is present.
    """
    return [k for k, i in enumerate(b_gaze_input) if i[1][0] > 0]


def create_tokenizer(bert_pretrained):
    """
    Wrapper function returning a tokenizer for BERT.
    """
    if bert_pretrained.startswith("xlm"):
        return XLMTokenizer.from_pretrained(bert_pretrained)
    else:
        return BertTokenizer.from_pretrained(bert_pretrained)


def n_modules(n, module, *args, **kwargs):
    """
    Creates n identical instances of module.
    """
    modules = []
    for _ in range(n):
        modules.append(module(*args, **kwargs))

    return modules


def save_json(obj, fpath):
    dir = os.path.dirname(fpath)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(fpath, "w") as f:
        json.dump(obj, f)


def load_json(fpath):
    with open(fpath) as f:
        return json.load(f)
