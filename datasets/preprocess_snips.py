
'''
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
This software is licensed under the BSD 3-Clause License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
PURPOSE.
'''

import os
from collections import defaultdict
import json
import csv

def read_csv_data(file_path):
    with open(file_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    asr_data = []
    raw_data = []
    first_line = True
    for raw, asr, _, _ in data:
        if first_line:
            first_line = False
            continue
        asr_data.append(asr)
        raw_data.append(raw)

    return raw_data, asr_data

if __name__ == '__main__':
    base_dir = "./snips_tts_asr/"
    output_dir = "SNIPS"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for split in ["train", "valid", "test"]:
        raw_data, asr_data = read_csv_data(base_dir + split + ".supconf.csv")

        save_dict = defaultdict(dict)
        for idx, (asr, raw) in enumerate(zip(asr_data, raw_data)):
            save_dict[idx + 1]["RAW"] = raw
            save_dict[idx + 1]["ASR"] = asr

        save_path = os.path.join(output_dir, split + ".json")
        with open(save_path, "w") as wfile:
            json.dump(save_dict, wfile, indent=4)
