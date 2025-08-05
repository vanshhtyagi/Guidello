# finetune_llama3_fixed.py
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
import torch
import os
import getpass

# -----------------------------------------------------------------------------
# 1. AUTHENTICATION
# -----------------------------------------------------------------------------
# Prefer a pre-exported environment variable. Fall back to an interactive prompt
# only when we are running in an interactive environment (e.g. notebook).
if "HF_TOKEN" not in os.environ:
    try:
        os.environ["HF_TOKEN"] = getpass.getpass(
            "🤗  Enter your Hugging Face token (will not be echoed): "
        )
    except Exception:
        raise RuntimeError(
            "HF_TOKEN environment variable not set and unable to request token interactively."
        )

# -----------------------------------------------------------------------------
# 2. DEVICE
# -----------------------------------------------------------------------------
# Use a torch.device object throughout instead of a bare string.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# 3. MODEL & TOKENIZER
# -----------------------------------------------------------------------------
# "meta-llama/Llama-3.2-3B-Instruct" is not (currently) a valid repo-id on the
# Hugging Face Hub. Replace it with the official name (update this if a smaller
# checkpoint becomes available).
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# Use the fast tokenizer implementation and add an explicit pad token because
# Llama does not provide one.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    quantization_config=quant_config,
    device_map="auto",
)
# Prepare the model for low-bit training *before* adding the LoRA adapters.
model = prepare_model_for_kbit_training(model)

# -----------------------------------------------------------------------------
# 4. PEFT (LoRA)
# -----------------------------------------------------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# -----------------------------------------------------------------------------
# 5. PROMPT BUILDING HELPERS
# -----------------------------------------------------------------------------

def extract_prompt_response(messages):
    """Convert list[dict(role, content)] -> llama-3 chat prompt."""
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
    assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), None)

    prompt = ""
    if system_msg:
        prompt += f"<|system|>\n{system_msg}\n"
    if user_msg:
        prompt += f"<|user|>\n{user_msg}\n"
    prompt += "<|assistant|>\n"
    return prompt, assistant_msg


def preprocess(example):
    prompt, response = extract_prompt_response(example["messages"])
    return {
        "text": prompt + (response or ""),
        "weight": float(example.get("weight", 1.0)),  # default weight=1.0
    }

# -----------------------------------------------------------------------------
# 6. DATASET
# -----------------------------------------------------------------------------
# We keep the `weight` column throughout the pipeline so that it is available in
# the data collator / Trainer.

dataset = load_dataset(
    "json",
    data_files={"train": "train_large.jsonl", "validation": "val_large.jsonl"},
)

text_datasets = dataset.map(
    preprocess,
    batched=False,
    remove_columns=[c for c in dataset["train"].column_names if c != "weight"],
)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=8192,  # Llama-3 supports an 8k context window.
        padding=False,
        return_tensors=None,
    )

# IMPORTANT: do NOT drop the `weight` column this time – we need it later.
tokenized_datasets = text_datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=[c for c in text_datasets["train"].column_names if c != "weight"],
    desc="Tokenising dataset",
)

# Enable torch formatting for efficient dataloader behaviour.
tokenized_datasets.set_format("torch")

train_weights = tokenized_datasets["train"]["weight"]
print(
    f"Training weight stats – min: {min(train_weights):.2f}, "
    f"max: {max(train_weights):.2f}, "
    f"avg: {sum(train_weights)/len(train_weights):.2f}"
)

# -----------------------------------------------------------------------------
# 7. DATA COLLATOR WITH PER-EXAMPLE WEIGHTS
# -----------------------------------------------------------------------------

class WeightedDataCollator(DataCollatorForLanguageModeling):
    """Same as standard LM collator but keeps a vector of per-item weights."""

    def __call__(self, features):
        weights = [f.pop("weight", 1.0) for f in features]
        batch = super().__call__(features)
        batch["weights"] = torch.tensor(weights, dtype=torch.float32).to(batch["input_ids"].device)
        return batch


# -----------------------------------------------------------------------------
# 8. WEIGHT-AWARE TRAINER
# -----------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """Trainer that scales the per-sample loss by a user-supplied weight vector."""

    def compute_loss(self, model, inputs, return_outputs=False):
        weights = inputs.pop("weights")  # shape: (batch,)
        labels = inputs["labels"]

        outputs = model(**inputs)
        logits = outputs.logits

        # Cross-entropy with no reduction => token-level losses
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        token_loss = token_loss.view(labels.shape)  # (batch, seq_len)

        # Average across tokens for each sample, then apply per-sample weights
        sample_loss = (token_loss.mean(dim=1) * weights).mean()

        return (sample_loss, outputs) if return_outputs else sample_loss


# -----------------------------------------------------------------------------
# 9. TRAINING ARGS & TRAINER
# -----------------------------------------------------------------------------

training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=202,
    eval_steps=202,
    output_dir="./qlora-llama3-8b-instruct",
    report_to="none",
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    gradient_checkpointing=True,
    optim="adamw_torch",
    warmup_steps=100,
    weight_decay=0.01,
    max_grad_norm=1.0,
    remove_unused_columns=False,
    dataloader_drop_last=True,
    group_by_length=True,
)

data_collator = WeightedDataCollator(tokenizer=tokenizer, mlm=False)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("🚀  Starting training …")
trainer.train()
print("✅  Training finished.")

# -----------------------------------------------------------------------------
# 10. SAVE CHECKPOINT (MERGED & DE-QUANTISED)
# -----------------------------------------------------------------------------

def save_dequantized(model, save_path):
    """Merge LoRA weights into the base model and save FP16 weights."""
    if isinstance(model, PeftModel):
        print("Merging LoRA weights …")
        model = model.merge_and_unload()
    model.to("cpu")
    model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)


save_dequantized(trainer.model, "./llama3-finetuned-fp16")