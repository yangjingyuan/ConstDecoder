
'''
Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
This software is licensed under the BSD 3-Clause License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
PURPOSE.
'''

from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import BertTokenizer
from data_reader import *

def further_convert_tags(src, tokenizer):
    wordpiece_token_lst = []
    for token_item in src:
        tmp_token_lst = tokenizer.tokenize(token_item)
        if len(tmp_token_lst) == 1:
            wordpiece_token_lst.append(tmp_token_lst[0])
        else:
            wordpiece_token_lst.extend(tmp_token_lst)
    return wordpiece_token_lst

class TagDecoder(nn.Module):
    def __init__(self, args):
        super(TagDecoder, self).__init__()
        #config
        self.args = args

        #encoder
        self.bert_encoder = BertModel.from_pretrained(args.base_model)
        self.tag_linear = nn.Sequential(nn.Dropout(args.tag_pdrop),
                                        nn.Linear(args.tag_hidden_size, args.tag_size))

        #decoder
        self.decoder_scale = nn.Sequential(nn.Dropout(args.tag_pdrop), \
                                           nn.Linear(768 * 2, 768))

        decoder_layer = nn.TransformerDecoderLayer(d_model=768, nhead=12, batch_first=True)
        self.edit_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.decoder_proj = nn.Sequential(nn.Linear(args.tag_hidden_size, args.vocab_size), \
                                          nn.Softmax(dim=-1))

        #generate
        self.simple_tokenizer = SimpleTokenizer(" ")
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args.tokenizer_name, \
                                              do_lower_case=True, do_basic_tokenize=False)

    def forward(self, inputs):
        src_token, src_mask, src_pos, target = inputs["src_token"].to(self.args.device), \
                                               inputs["src_mask"].to(self.args.device), \
                                               inputs["src_pos"].to(self.args.device), \
                                               inputs["target"].to(self.args.device)

        tag_labels = target[..., 0]
        batch_size, src_len, max_added_len = target.size()
        target_mask = target.bool()

        #encoder
        bert_outputs = self.bert_encoder(input_ids=src_token, attention_mask=src_mask,\
                            token_type_ids=None, position_ids=src_pos)

        src_encode = bert_outputs.last_hidden_state

        edit_tags = self.tag_linear(src_encode)

        tag_weight = torch.ones(self.args.tag_size, dtype=torch.float, device=self.args.device)
        tag_weight[-1] = self.args.change_weight

        tag_loss = F.cross_entropy(edit_tags.view(-1, self.args.tag_size), tag_labels.view(-1),
                                   weight=tag_weight)

        #decoder
        if target.size(-1) == 1:
            gen_loss = torch.tensor(0)
        elif target.size(-1) <= 3:
            raise ValueError("Check the preprocess, must be the form of '[CLS] ... [SEP]' ")
        else:
            batch_memory = []
            batch_memory_mask = []
            batch_changes = []

            batch_gen_input_ids = []
            batch_gen_labels = []
            batch_gen_mask = []

            for batch_id in range(batch_size):
                for token_id in range(src_len):
                    if target[batch_id][token_id][0] != 2: #if not change
                        continue
                    else:
                        batch_memory.append(src_encode[batch_id].unsqueeze(0))
                        batch_memory_mask.append(src_mask[batch_id].unsqueeze(0))
                        batch_changes.append(src_encode[batch_id][token_id].unsqueeze(0))

                        batch_gen_input_ids.append(target[batch_id][token_id][1:-1].unsqueeze(0))
                        batch_gen_mask.append(target_mask[batch_id][token_id][1:-1].unsqueeze(0))
                        batch_gen_labels.append(target[batch_id][token_id][2:].unsqueeze(0))

            batch_memory = torch.cat(batch_memory, dim=0)
            batch_memory_mask = ~torch.cat(batch_memory_mask, dim=0).bool()

            batch_changes = torch.cat(batch_changes, dim=0)
            batch_gen_input_ids = torch.cat(batch_gen_input_ids, dim=0)
            batch_gen_labels = torch.cat(batch_gen_labels, dim=0)

            batch_gen_mask = ~torch.cat(batch_gen_mask, dim=0).bool()

            gen_input_embeds = self.bert_encoder.embeddings(batch_gen_input_ids)
            batch_changes = batch_changes.unsqueeze(1).repeat(1, gen_input_embeds.size(1), 1)

            fused_gen_input = torch.cat([gen_input_embeds, batch_changes], -1)
            fused_gen_input = self.decoder_scale(fused_gen_input)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(fused_gen_input.size(1))
            decoder_outputs = self.edit_decoder(tgt=fused_gen_input, \
                                                memory=batch_memory, \
                                                tgt_mask=tgt_mask.to(self.args.device), \
                                                memory_mask=None, \
                                                tgt_key_padding_mask=batch_gen_mask, \
                                                memory_key_padding_mask=batch_memory_mask)

            gen_probs = self.decoder_proj(decoder_outputs)
            gen_loss = F.nll_loss(gen_probs.log().view(-1, self.args.vocab_size),
                                  batch_gen_labels.view(-1), ignore_index=self.args.pad_token_id)

        total_loss = self.args.alpha * tag_loss + gen_loss
        return tag_loss, gen_loss, total_loss

    def generate(self, input_str):
        src = insert_dummy(self.simple_tokenizer.tokenize(input_str))
        src = further_convert_tags(src, self.bert_tokenizer)

        src = self.bert_tokenizer.convert_tokens_to_ids(src)
        src = [self.bert_tokenizer.cls_token_id] + src + [self.bert_tokenizer.sep_token_id]
        src_mask = [1] * len(src)
        src_pos = [x for x in range(len(src))]
        inputs = {}

        inputs["src_token"] = torch.tensor(src, dtype=torch.long)
        inputs["src_mask"] = torch.tensor(src_mask, dtype=torch.long)
        inputs["src_pos"] = torch.tensor(src_pos, dtype=torch.long)

        src_token, src_mask, src_pos = inputs["src_token"].to(self.args.device), \
                                       inputs["src_mask"].to(self.args.device), \
                                       inputs["src_pos"].to(self.args.device)
        #encoder
        src_token = src_token.unsqueeze(0)
        src_mask = src_mask.unsqueeze(0)
        src_pos = src_pos.unsqueeze(0)

        batch_size, src_len = src_token.size()
        # encoder
        bert_outputs = self.bert_encoder(input_ids=src_token, attention_mask=src_mask, \
                                         token_type_ids=None, position_ids=src_pos)

        src_encode = bert_outputs.last_hidden_state
        edit_tags = self.tag_linear(src_encode)
        tag_pred = torch.argmax(edit_tags, dim=-1)

        #decoder
        batch_id_token_id2count_id = {}
        cnt = 0
        batch_memory = []
        batch_changes = []

        for batch_id in range(batch_size):
            for token_id in range(src_len):
                if tag_pred[batch_id][token_id] != 2:
                    continue
                else:
                    batch_id_token_id2count_id[(batch_id, token_id)] = cnt
                    cnt += 1
                    batch_memory.append(src_encode[batch_id].unsqueeze(0))
                    batch_changes.append(src_encode[batch_id][token_id].unsqueeze(0))

        pred_tags = tag_pred[0].tolist()
        ori_tokens = src_token[0].tolist()
        if not batch_id_token_id2count_id: #empty dict
            output_str = []
            for pred_tag, ori_token in zip(pred_tags, ori_tokens):
                if pred_tag == 1: #keep symbol
                    output_str.append(ori_token)

            return self.bert_tokenizer.decode(output_str)
        else:
            batch_memory = torch.cat(batch_memory, dim=0)
            batch_changes = torch.cat(batch_changes, dim=0).unsqueeze(1)

            #start symbol 101
            batch_gen_input_ids = torch.ones(batch_changes.size(0), 1).fill_(self.bert_tokenizer.cls_token_id).type(torch.long).to(self.args.device)
            for i in range(self.args.max_add_len - 1):
                gen_input_embeds = self.bert_encoder.embeddings(batch_gen_input_ids)
                batch_changes_new = batch_changes.repeat(1, gen_input_embeds.size(1), 1) #same length
                fused_gen_input = torch.cat([gen_input_embeds, batch_changes_new], -1)
                fused_gen_input = self.decoder_scale(fused_gen_input)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(fused_gen_input.size(1))
                decoder_outputs = self.edit_decoder(tgt=fused_gen_input, \
                                                    memory=batch_memory, \
                                                    tgt_mask=tgt_mask.to(self.args.device))

                gen_probs = self.decoder_proj(decoder_outputs)
                next_word = torch.argmax(gen_probs, dim=-1)

                batch_gen_input_ids = torch.cat([batch_gen_input_ids, next_word[:,-1].unsqueeze(1)], dim=1)

            batch_gen_input_ids = batch_gen_input_ids.tolist()
            for k, v in batch_id_token_id2count_id.items():
                end_pos = batch_gen_input_ids[v].index(self.bert_tokenizer.sep_token_id) \
                    if self.bert_tokenizer.sep_token_id in batch_gen_input_ids[v] else -1
                gen_con = batch_gen_input_ids[v][1:end_pos]
                batch_id_token_id2count_id[k] = gen_con

            # #relax
            result = []
            for pidx, (pred_tag, token) in enumerate(zip(pred_tags, ori_tokens)):
                if pred_tag == 0:
                    continue
                elif pred_tag == 1:
                    result.append(token)
                elif pred_tag == 2:
                    result.extend(batch_id_token_id2count_id[(0, pidx)])

            return self.bert_tokenizer.decode(result, skip_special_tokens=True)
