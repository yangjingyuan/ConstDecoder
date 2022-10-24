
'''
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
This software is licensed under the BSD 3-Clause License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
PURPOSE.
'''

import argparse
import time
import json
from jiwer import wer
from model import *

def read_data(data_path):
    with open(data_path) as rfile:
        data = json.load(rfile)

    src = []
    tgt = []
    for k, v in data.items():
        src.append(v["ASR"] if v["ASR"] != None else "")
        tgt.append(v["RAW"] if v["RAW"] != None else "")

    new_data = {}
    new_data["source"] = src
    new_data["target"] = tgt
    return new_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--base_model", type=str, default="", help="")
    parser.add_argument("--device", type=str, default="", help="")
    parser.add_argument("--tag_pdrop", type=float, default=0.2, help="")
    parser.add_argument("--decoder_proj_pdrop", type=float, default=0.2, help="")
    parser.add_argument("--tag_hidden_size", type=int, default=768, help="")
    parser.add_argument("--tag_size", type=int, default=3, help="")
    parser.add_argument("--change_weight", type=float, default=3.0, help="")
    parser.add_argument("--vocab_size", type=int, default=30522, help="")
    parser.add_argument("--pad_token_id", type=int, default=0, help="")
    parser.add_argument("--alpha", type=float, default=3.0, help="")
    parser.add_argument("--tokenizer_name", type=str, default="", help="")
    parser.add_argument("--max_add_len", type=int, default=10, help="")
    parser.add_argument("--model_path", type=str, default="", help="")
    parser.add_argument("--test_data_path", type=str, default="", help="")
    args = parser.parse_args()
    args.device = "cuda:" + args.device

    #define model
    model = TagDecoder(args)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model = model.to(args.device)
    model.eval()

    #read data
    data = read_data(args.test_data_path)
    total_start_time = time.time()
    sample_num = 0
    preds = []
    for asr_str, gold in zip(data["source"], data["target"]):
        start_time = time.time()
        pred = model.generate(asr_str)
        end_time = time.time()
        pred_time = (end_time - start_time)
        preds.append(pred.strip())
        print(f"ASR:{asr_str}")
        print(f"PRED:{pred}")
        print(f"GOLD:{gold}")
        sample_num += 1

    total_end_time = time.time()
    print(f"avg time cost is {(total_end_time - total_start_time) / sample_num}")
    print("raw_wer: {}\nnew_wer: {}\n".format(wer(data["target"], data["source"]), wer(data["target"], preds)))
