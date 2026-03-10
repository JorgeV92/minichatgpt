# MiniChatGPT

A small, **GPT-style chatbot**.

GPT-style chatbot built from scratch in `python` and `Pytorch` 

## What this project includes

- a **byte-level BPE tokenizer** implemented from scratch
- a **decoder-only Transformer** for autoregressive language modeling
- a **pretraining pipeline** for next-token prediction on raw text
- TODO a **chat supervised fine-tuning pipeline** on instruction-response JSONL data
- TODO: a **terminal chat interface** for talking to the model
<!-- - an optional **Gradio web chat UI** for a more ChatGPT-like demo -->

## 1. Train a tokenizer

```bash
python scripts/train_tokenizer.py \
  --input data/tiny_corpus.txt \
  --output artifacts/tokenizer.json \
  --vocab-size 320
```

## 2. Pretrain a tiny GPT

```bash
python scripts/pretrain.py \
  --input data/tiny_corpus.txt \
  --tokenizer artifacts/tokenizer.json \
  --out-dir artifacts/pretrain \
  --epochs 20 \
  --batch-size 8
```

## 3. Fine-tune for chat

```bash
python scripts/finetune_chat.py \
  --data data/sample_chat.jsonl \
  --tokenizer artifacts/tokenizer.json \
  --checkpoint artifacts/pretrain/best_model.pt \
  --out-dir artifacts/chat \
  --epochs 30 \
  --batch-size 
```


## 4. Chat in the terminal

```bash
python scripts/chat.py \
  --checkpoint artifacts/chat/best_model.pt \
  --tokenizer artifacts/tokenizer.json
```

The current set up works but the training is mininal and the output of llm is just nonsense when working in terminal UI. TODO: train the llm so it could have a basic conversation with a user.

```bash
minichatgpt % python scripts/chat.py \
  --checkpoint artifacts/chat/best_model.pt \
  --tokenizer artifacts/tokenizer.json
MiniChatGPT terminal chat
Type 'exit' to quit.

you> Hello
bot> �i sst fver ae ktseoe sl yoneskens ase ah tu ae a tfyl rkeninrqmd n ss inspfisiesmineiens a ie e e qe eu-n

you> Not so great huh!
bot> Athen solhsf.ie  ty o ntoodel q aeeensky k t aie kolifos codel sese kouratsho.se s a s tst tmts n kenss

you> exit
bye
```