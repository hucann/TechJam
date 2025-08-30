# pip install transformers==4.42.0 trl peft bitsandbytes accelerate datasets pandas jsonschema

import os, json
import numpy as np
import pandas as pd
from datasets import Dataset
import torch

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ------------------ CONFIG ------------------
MODEL_NAME = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
CSV_PATH   = "/home/xinyun/code/llmtrain/optimized_classified_reviews_labelled.csv"
OUT_DIR    = os.getenv("OUT_DIR", "out/llm-multilabel-qlora")
MAX_LEN    = 4000
EPOCHS     = 2
BATCH      = 2
GRAD_ACC   = 1
LR         = 1e-4
WARMUP     = 0.03

SYSTEM_PROMPT = (
    "You are a strict policy classifier for restaurant reviews. "
    "Return ONLY valid JSON with four booleans:\n"
    "{\n"
    '  "advertisement_violation": true|false,\n'
    '  "irrelevant_violation": true|false,\n'
    '  "rant_violation": true|false,\n'
    '  "overall_compliant": true|false\n'
    "}\n"
    "Definitions:\n"
    "- advertisement_violation: ads/promos/URLs/phones/emails/coupon codes.\n"
    "- irrelevant_violation: not about this venue (off-topic/other place/products).\n"
    "- rant_violation: generic rant likely not from a real visit (no specifics).\n"
    "- overall_compliant: no violations.\n"
    "Reply with JSON only."
)

# ------------------ DATA ------------------
def to_bool(x):
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if x is None or (isinstance(x, float) and np.isnan(x)): return False
    s = str(x).strip().lower()
    return s in {"1","true","yes","y","t"}

def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    needed = ["text","name",
              "advertisement_violation","irrelevant_violation","rant_violation","overall_compliant"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # normalize
    df["text"] = df["text"].fillna("").astype(str)
    df["name"] = df["name"].fillna("").astype(str)
    for c in ["advertisement_violation","irrelevant_violation","rant_violation","overall_compliant"]:
        df[c] = df[c].apply(to_bool)

    # enforce consistency (if labels were noisy)
    v_any = df[["advertisement_violation","irrelevant_violation","rant_violation"]].any(axis=1)
    df["overall_compliant"] = (~v_any)

    # Split (simple): stratify by 'any violation' to balance
    df["any_violation"] = v_any.astype(int)
    train = df.sample(frac=0.9, random_state=42)
    valid = df.drop(train.index)

    return train.drop(columns=["any_violation"]), valid.drop(columns=["any_violation"])

def build_messages(row):
    user = (
        f"[REVIEW]\n{row['text']}\n\n"
        f"[PLACE]\nName: {row.get('name','')}"
    )
    labels = {
        "advertisement_violation": bool(row["advertisement_violation"]),
        "irrelevant_violation": bool(row["irrelevant_violation"]),
        "rant_violation": bool(row["rant_violation"]),
        "overall_compliant": bool(row["overall_compliant"]),
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(labels, separators=(",",":"))}
        ]
    }

# ------------------ MODEL ------------------
def load_model_and_tok():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tok.pad_token = tok.eos_token
    if tok.chat_template is None:
        tok.chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif (message['role'] == 'system') %}{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}{% elif (message['role'] == 'assistant') %}{{ ' ' + message['content'] + ' ' + eos_token }}{% endif %}{% endfor %}"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=True,
        device_map="auto",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_cfg)
    return model, tok

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train_df, val_df = load_and_clean(CSV_PATH)

    train_ds = Dataset.from_list([build_messages(r) for _, r in train_df.iterrows()])
    val_ds   = Dataset.from_list([build_messages(r) for _, r in val_df.iterrows()])

    model, tok = load_model_and_tok()

    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP,
        logging_steps=20,
        save_steps=5000,
        evaluation_strategy="steps",
        eval_steps=5000,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=8,
        report_to="none",
        max_grad_norm=1.0
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        eval_dataset=val_ds,        
        formatting_func=lambda example: tok.apply_chat_template(example['messages'], tokenize=False),
        max_seq_length=MAX_LEN,
        packing=False,
        args=args
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)

if __name__ == "__main__":
    main()