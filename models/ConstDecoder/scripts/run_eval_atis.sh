
PRETRAINED_MODEL=bert-base-uncased
nohup python -u ../eval.py \
    --base_model $PRETRAINED_MODEL \
    --tokenizer_name $PRETRAINED_MODEL \
    --model_path ./models.atis/best.pt \
    --test_data_path ../../../datasets/ATIS/test.json \
    --device 0 \
    --tag_pdrop 0.2 \
    --decoder_proj_pdrop 0.2 \
    --tag_hidden_size 768 \
    --tag_size 3 \
    --alpha 3.0 \
    --change_weight 1.5 \
    --vocab_size 30522 \
    --pad_token_id 0 \
    --max_add_len 10 >eval_atis.log 2>&1 &