
'''
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
This software is licensed under the BSD 3-Clause License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
PURPOSE.
'''

import os
import zipfile
from collections import defaultdict
import json


def unzip_zip(from_path, to_path):
    with zipfile.ZipFile(from_path, 'r') as zip_ref:
        zip_ref.extractall(to_path)

def read_file(file_path):
    data = []
    with open(file_path) as rfile:
        for line in rfile:
            data.append(line.strip())

    return data

if __name__ == '__main__':
    unzip_zip("atis.zip", "atis")

    base_dir = "./atis/atis/"
    output_dir = "ATIS"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for split in ["train", "valid", "test"]:
        asr_file = base_dir + split + "/en.classification.txt"
        raw_file = base_dir + split + "/text_original.txt"
        asr_data = read_file(asr_file)
        raw_data = read_file(raw_file)

        save_dict = defaultdict(dict)
        for idx, (asr, raw) in enumerate(zip(asr_data, raw_data)):
            save_dict[idx + 1]["RAW"] = raw
            save_dict[idx + 1]["ASR"] = asr

        save_path = os.path.join(output_dir, split + ".json")
        with open(save_path, "w") as wfile:
            json.dump(save_dict, wfile, indent=4)
