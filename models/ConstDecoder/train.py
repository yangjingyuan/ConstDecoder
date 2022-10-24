
'''
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
This software is licensed under the BSD 3-Clause License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
PURPOSE.
'''

import os
import shutil
from torch.optim import Adam
from jiwer import wer
import argparse
from data_reader import *
from model import *


def train_epoch(model, train_data_loader, optimizer):
    loss_list = []
    model.train()
    for i, batch in enumerate(train_data_loader):
        optimizer.zero_grad()
        tag_loss, gen_loss, total_loss = model(batch)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_list.append(total_loss)
        if i % 20 == 0:  # monitoring
            print(f"train step: {i}, tag loss is {tag_loss.item()}, gen loss is {gen_loss.item()}, total loss is {total_loss.item()}")

    return sum(loss_list) / len(loss_list)

def valid_epoch(model, valid_data_loader):
    model.eval()
    preds = []
    golds = []
    with torch.no_grad():
        for i, batch in enumerate(valid_data_loader):
            raws = batch["RAWS"]
            asrs = batch["ASRS"]
            for asr, raw in zip(asrs, raws):
                pred = model.generate(asr)
                preds.append(pred.strip())
                golds.append(raw)

    return wer(golds, preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    #model
    parser.add_argument("--base_model", type=str, default="", help="")
    parser.add_argument("--tag_pdrop", type=float, default=0.2, help="")
    parser.add_argument("--decoder_proj_pdrop", type=float, default=0.2, help="")
    parser.add_argument("--tag_hidden_size", type=int, default=768, help="")
    parser.add_argument("--tag_size", type=int, default=3, help="")
    parser.add_argument("--vocab_size", type=int, default=30522, help="")
    parser.add_argument("--pad_token_id", type=int, default=0, help="")
    parser.add_argument("--alpha", type=float, default=3.0, help="")
    parser.add_argument("--change_weight", type=float, default=1.5, help="")

    #data
    parser.add_argument("--train_data_file", type=str, default="", help="")
    parser.add_argument("--eval_data_file", type=str, default="", help="")
    parser.add_argument("--max_src_len", type=int, default=256, help="")
    parser.add_argument("--max_add_len", type=int, default=10, help="")
    parser.add_argument("--tokenizer_name", type=str, default="", help="")

    #train
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--lr", type=float, default=5e-5, help="")
    parser.add_argument("--max_num_epochs", type=int, default=10, help="")
    parser.add_argument("--save_dir", type=str, default="", help="")
    parser.add_argument("--device", type=str, default="", help="")

    args = parser.parse_args()
    args.device = "cuda:" + args.device

    #load data
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name, \
                                              do_lower_case=True, do_basic_tokenize=False)
    train_examples = get_examples(examples_path=args.train_data_file,
                                  tokenizer=tokenizer,
                                  max_src_len=args.max_src_len,
                                  max_add_len=args.max_add_len,
                                  )
    eval_examples = get_examples(examples_path=args.eval_data_file,
                                 tokenizer=tokenizer,
                                 max_src_len=args.max_src_len,
                                 max_add_len=args.max_add_len,
                                 )

    train_dataset = ExampleDataset(train_examples)
    valid_dataset = ExampleDataset(eval_examples)
    train_data_loader = DataLoader(train_dataset, collate_fn=collate_batch, batch_size=args.batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, collate_fn=collate_batch, batch_size=args.batch_size, shuffle=True)


    #define model, loss_fn, optimizer
    model = TagDecoder(args)
    model = model.to(args.device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    eval_loss_list = []
    for epoch in range(1, args.max_num_epochs + 1):
        print(f"=========train at epoch={epoch}=========")
        avg_train_loss = train_epoch(model, train_data_loader, optimizer)
        print(f"train {epoch} average loss is {avg_train_loss}")

        print(f"=========eval at epoch={epoch}=========")
        avg_val_loss = valid_epoch(model, valid_data_loader)
        torch.save(model.state_dict(), args.save_dir + f"/{epoch}.pt")
        print(f"eval {epoch} wer is {avg_val_loss}")
        eval_loss_list.append((epoch, avg_val_loss))

    eval_loss_list.sort(key=lambda x:x[-1])
    print(eval_loss_list)
    best_epoch_path = os.path.join(args.save_dir, str(eval_loss_list[0][0]) + ".pt")
    print(f"best epoch path is {best_epoch_path}")
    shutil.copyfile(best_epoch_path, os.path.join(args.save_dir, f"best.pt"))
