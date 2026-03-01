"""
train_qlora.py
==============
QLoRA fine-tuning of OpenBioLLM-70B on MedCaseReason QA dataset.
- Optimizer state saved and reloaded between runs (continuous training)
- Runs sequentially through all 14k rows across multiple sessions

Hardware : 2x NVIDIA A100-PCIE-40GB (cross-NUMA, no NVLink)
Model    : aaditya/Llama3-OpenBioLLM-70B
Dataset  : medcasereasoning_core.csv (Stanford MedCaseReason QA)
Versions : transformers==4.44.2, accelerate==0.33.0, bitsandbytes==0.49.1, torch==2.6.0

Run in tmux:
    CUDA_VISIBLE_DEVICES=0,1 python train_qlora.py 2>&1 | tee -a train_stdout.log

Monitor:
    tail -f /media/rvcse22/CSERV/medlm/train.log
    watch -n 30 cat /media/rvcse22/CSERV/medlm/loss_log.csv

HOW TO PROGRESS THROUGH ALL 14k ROWS:
    Run 1 : TRAIN_SUBSET = 2000   (already done)
    Run 2 : TRAIN_SUBSET = 4000   → trains rows 2000-3999
    Run 3 : TRAIN_SUBSET = 6000   → trains rows 4000-5999
    Run 4 : TRAIN_SUBSET = 8000   → trains rows 6000-7999
    Run 5 : TRAIN_SUBSET = 10000  → trains rows 8000-9999
    Run 6 : TRAIN_SUBSET = 12000  → trains rows 10000-11999
    Run 7 : TRAIN_SUBSET = 14489  → trains rows 12000-14488
"""

# ── Environment (must be before any torch import) ──────────────────────────────
import os
os.environ["CUDA_VISIBLE_DEVICES"]     = "0,1"
os.environ["TRANSFORMERS_OFFLINE"]    = "1"
os.environ["HF_DATASETS_OFFLINE"]     = "1"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Imports ────────────────────────────────────────────────────────────────────
import csv
import time
import math
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — only edit TRAIN_SUBSET each run
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH = (
    "/media/rvcse22/CSERV/medlm/"
    "models--aaditya--Llama3-OpenBioLLM-70B/snapshots/"
    "7ad17ef0d2185811f731f89d20885b2f99b1e994"
)
DATA_PATH  = "/media/rvcse22/CSERV/medlm/medcasereasoning_core.csv"
OUTPUT_DIR = "/media/rvcse22/CSERV/medlm/qlora-medical"
LOG_FILE   = "/media/rvcse22/CSERV/medlm/train.log"
LOSS_CSV   = "/media/rvcse22/CSERV/medlm/loss_log.csv"

# ── Dataset ────────────────────────────────────────────────────────────────────
TRAIN_SUBSET   = 13100
MAX_LENGTH     = 1024

# ── Training ───────────────────────────────────────────────────────────────────
NUM_EPOCHS     = 1
BATCH_SIZE     = 1
GRAD_ACCUM     = 16
LEARNING_RATE  = 2e-4
WARMUP_STEPS   = 30
SAVE_STEPS     = 5
LOGGING_STEPS  = 5
SAVE_TOTAL     = 3

# ── LoRA ───────────────────────────────────────────────────────────────────────
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
LORA_TARGET   = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
# ══════════════════════════════════════════════════════════════════════════════

# ── Paths ──────────────────────────────────────────────────────────────────────
ADAPTER_DIR   = f"{OUTPUT_DIR}/adapter"
OPTIMIZER_DIR = f"{OUTPUT_DIR}/optimizer_state"
PROGRESS_FILE = f"{OUTPUT_DIR}/progress.json"

# ── Logging ────────────────────────────────────────────────────────────────────
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(LOG_FILE)
file_handler.stream = open(LOG_FILE, "a", buffering=1)  # line buffered = flush every line
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        file_handler,
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert clinical physician. You will be given a medical case.
Analyze it and respond in exactly this structure:

DIAGNOSTIC REASONING:
Step by step clinical reasoning through the case findings, symptoms, and lab values.
Be thorough — cover all relevant clinical findings, pathophysiology, and reasoning steps.

DIFFERENTIAL DIAGNOSIS:
List 3-5 alternative diagnoses with brief reasoning for why each is considered or excluded.

FINAL DIAGNOSIS:
State the final diagnosis clearly."""


# ── Progress tracking ──────────────────────────────────────────────────────────
def load_progress() -> dict:
    if Path(PROGRESS_FILE).exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"rows_trained": 0, "runs": []}


def save_progress(rows_trained: int, run_info: dict) -> None:
    progress = load_progress()
    progress["rows_trained"] = rows_trained
    progress["runs"].append(run_info)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)
    log.info(f"Progress saved: {rows_trained} rows trained total")


# ── Loss CSV callback ──────────────────────────────────────────────────────────
class LossLoggerCallback(TrainerCallback):
    """Logs step, loss, lr to CSV every LOGGING_STEPS for plotting later."""
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        if not Path(csv_path).exists():
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["global_step", "loss", "learning_rate", "epoch", "timestamp"])

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs and "loss" in logs:
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    state.global_step,
                    round(logs["loss"], 6),
                    round(logs.get("learning_rate", 0), 8),
                    round(state.epoch, 4),
                    datetime.now().isoformat(),
                ])


# ── Optimizer save/load callback ───────────────────────────────────────────────
class OptimizerCheckpointCallback(TrainerCallback):
    """Saves optimizer state to a fixed directory after every checkpoint save."""
    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Trainer already saved optimizer.pt in the checkpoint dir
        # We copy it to a fixed location so next run can load it
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        optimizer_src  = f"{checkpoint_dir}/optimizer.pt"
        scheduler_src  = f"{checkpoint_dir}/scheduler.pt"

        if Path(optimizer_src).exists():
            Path(OPTIMIZER_DIR).mkdir(parents=True, exist_ok=True)
            shutil.copy2(optimizer_src, f"{OPTIMIZER_DIR}/optimizer.pt")
            shutil.copy2(scheduler_src, f"{OPTIMIZER_DIR}/scheduler.pt")
            log.info(f"Optimizer state backed up at step {state.global_step}")


# ── Dataset ────────────────────────────────────────────────────────────────────
def format_sample(row: pd.Series) -> str:
    user_content      = row["case_prompt"].strip()
    assistant_content = (
        f"DIAGNOSTIC REASONING:\n{row['diagnostic_reasoning'].strip()}\n\n"
        f"FINAL DIAGNOSIS:\n{row['final_diagnosis'].strip()}"
    )
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n{assistant_content}<|eot_id|>"
    )


class MedCaseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = MAX_LENGTH):
        self.samples = []
        log.info(f"Tokenizing {len(df)} samples...")
        skipped = 0
        for _, row in df.iterrows():
            text = format_sample(row)
            enc  = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            if len(enc["input_ids"]) < 10:
                skipped += 1
                continue
            enc["labels"] = enc["input_ids"].copy()
            self.samples.append(enc)
        log.info(f"Dataset ready: {len(self.samples)} samples ({skipped} skipped)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── GPU utils ──────────────────────────────────────────────────────────────────
def log_gpu_stats(label: str = "") -> None:
    if label:
        log.info(f"[GPU] {label}")
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        used = total - free
        log.info(
            f"  cuda:{i} {torch.cuda.get_device_name(i)} | "
            f"Used: {used/1e9:.2f}GB / {total/1e9:.2f}GB ({used/total*100:.1f}%)"
        )


# ── Checkpoint helper ──────────────────────────────────────────────────────────
def get_last_checkpoint(output_dir: str):
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    checkpoints = sorted(
        [d for d in output_path.iterdir() if d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[-1])
    )
    if checkpoints:
        last = str(checkpoints[-1])
        log.info(f"Found checkpoint: {last}")
        return last
    return None


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    run_start    = time.time()
    progress     = load_progress()
    rows_already = progress["rows_trained"]
    rows_end     = TRAIN_SUBSET

    log.info("=" * 70)
    log.info("  QLoRA Fine-tuning — OpenBioLLM-70B — MedCaseReason QA")
    log.info(f"  Started      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Rows already : {rows_already}")
    log.info(f"  Rows this run: {rows_already} → {rows_end}")
    log.info(f"  LoRA         : r={LORA_R}, alpha={LORA_ALPHA}")
    log.info("=" * 70)

    if rows_already >= rows_end:
        log.info(f"Already trained up to row {rows_already}. Increase TRAIN_SUBSET to continue.")
        return

    # ── Load data (only new rows) ──────────────────────────────────────────────
    log.info(f"Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    log.info(f"Full dataset: {df.shape}")

    if "split" in df.columns:
        df = df[df["split"] == "train"].reset_index(drop=True)

    train_df = df.iloc[rows_already:rows_end].reset_index(drop=True)
    log.info(f"Training on rows {rows_already} to {rows_end} ({len(train_df)} new rows)")

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Dataset ────────────────────────────────────────────────────────────────
    train_dataset = MedCaseDataset(train_df, tokenizer)
    total_steps   = math.ceil(len(train_dataset) / (BATCH_SIZE * GRAD_ACCUM)) * NUM_EPOCHS
    log.info(f"Total steps this run : {total_steps}")
    log.info(f"Effective batch size : {BATCH_SIZE * GRAD_ACCUM}")

    # ── Model ──────────────────────────────────────────────────────────────────
    log.info("Loading base model in 4-bit NF4...")
    log_gpu_stats("Before model load")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
        max_memory={0: "36GiB", 1: "38GiB"},
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    log_gpu_stats("After model load")

    # ── Prepare for QLoRA ──────────────────────────────────────────────────────
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # Load existing adapter if not first run
    if Path(ADAPTER_DIR).exists() and rows_already > 0:
        log.info(f"Loading existing LoRA adapter from {ADAPTER_DIR}...")
        model = PeftModel.from_pretrained(model, ADAPTER_DIR, is_trainable=True)
        log.info("Adapter loaded — continuing from previous run")
    else:
        log.info("Initializing fresh LoRA weights...")
        model = get_peft_model(model, LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=LORA_TARGET,
        ))

    model.print_trainable_parameters()

    # ── Check for optimizer state from previous run ────────────────────────────
    optimizer_path = f"{OPTIMIZER_DIR}/optimizer.pt"
    scheduler_path = f"{OPTIMIZER_DIR}/scheduler.pt"
    has_optimizer  = Path(optimizer_path).exists() and rows_already > 0

    if has_optimizer:
        log.info("Found saved optimizer state — will inject after trainer init")
    else:
        log.info("No optimizer state found — starting fresh optimizer")

    # ── Resume from checkpoint within this run (crash recovery) ───────────────
    resume_from = get_last_checkpoint(OUTPUT_DIR)

    # ── Training args ──────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,
        ),
        callbacks=[
            LossLoggerCallback(LOSS_CSV),
            OptimizerCheckpointCallback(),
        ],
    )

    # ── Inject optimizer state from previous run ───────────────────────────────
    if has_optimizer and resume_from is None:
        log.info("Injecting optimizer state from previous run...")
        try:
            trainer.create_optimizer_and_scheduler(num_training_steps=total_steps)
            opt_state  = torch.load(optimizer_path, map_location="cpu")
            sch_state  = torch.load(scheduler_path, map_location="cpu")
            trainer.optimizer.load_state_dict(opt_state)
            trainer.lr_scheduler.load_state_dict(sch_state)
            log.info("Optimizer state loaded successfully — smooth continuation")
        except Exception as e:
            log.warning(f"Could not load optimizer state: {e} — starting fresh")

    # ── Train ──────────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("Training started...")
    log.info("=" * 70)
    log_gpu_stats("Before training")

    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save adapter ───────────────────────────────────────────────────────────
    log.info(f"Saving LoRA adapter to {ADAPTER_DIR}...")
    Path(ADAPTER_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    elapsed = time.time() - run_start

    # Save metadata alongside adapter
    metadata = {
        "model":           MODEL_PATH,
        "dataset":         DATA_PATH,
        "rows_this_run":   len(train_dataset),
        "rows_start":      rows_already,
        "rows_end":        rows_end,
        "total_rows":      14489,
        "epochs":          NUM_EPOCHS,
        "total_steps":     total_steps,
        "lora_r":          LORA_R,
        "lora_alpha":      LORA_ALPHA,
        "max_length":      MAX_LENGTH,
        "learning_rate":   LEARNING_RATE,
        "elapsed_hours":   round(elapsed / 3600, 2),
        "completed":       datetime.now().isoformat(),
    }
    with open(f"{ADAPTER_DIR}/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save progress for next run
    save_progress(rows_end, metadata)

    log.info("=" * 70)
    log.info(f"Run complete! Elapsed : {round(elapsed/3600, 2)} hours")
    log.info(f"Rows trained so far  : {rows_end} / 14489")
    log.info(f"Remaining rows       : {14489 - rows_end}")
    if rows_end < 14489:
        log.info(f"Next run             : set TRAIN_SUBSET = {min(rows_end + 2000, 14489)}")
    else:
        log.info("All 14489 rows complete!")
    log.info(f"Loss log             : {LOSS_CSV}")
    log.info("=" * 70)
    log_gpu_stats("After training")


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT LOSS — save as plot_loss.py and run on any machine after training
#
# import pandas as pd
# import matplotlib.pyplot as plt
#
# df = pd.read_csv("/media/rvcse22/CSERV/medlm/loss_log.csv")
# plt.figure(figsize=(14, 5))
# plt.plot(df["global_step"], df["loss"], linewidth=1.5, color="steelblue")
# plt.xlabel("Global Step")
# plt.ylabel("Training Loss")
# plt.title("OpenBioLLM-70B QLoRA — MedCaseReason QA (Stanford)")
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("loss_curve.png", dpi=150)
# print("Saved loss_curve.png")
# ══════════════════════════════════════════════════════════════════════════════
