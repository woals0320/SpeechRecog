from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer와 model 정의
tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
model = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-base-v2")

import shutil
shutil.rmtree("kobart_gec_with_freq_mistake_v2", ignore_errors=True)

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    logging
)

# 로그 레벨 설정
logging.set_verbosity_info()

ds = load_dataset(
    "csv",
    data_files={
        "train": "./datasets/kobart_train+freq.tsv",
        "validation": "./datasets/kobart_valid+freq.tsv"
    },
    delimiter="\t",
)

def preprocess(examples):
    inputs = [f"{s}" for s in examples["source"]]  # 또는 f"interface:... {s}" 로 붙여도 됨
    targets = [t if isinstance(t, str) else "" for t in examples["target"]]

    model_inputs = tokenizer(
        inputs,
        max_length=64,
        padding="max_length",
        truncation=True
    )
    labels = tokenizer(
        text_target=targets,
        max_length=64,
        padding="max_length",
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = ds.map(
    preprocess,
    batched=True,
    remove_columns=["source", "target"]
)

# Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 학습 인자 설정
training_args = TrainingArguments(
    output_dir="train_4_20250613_0149",
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=200,
    logging_dir="train_4_20250613_0149/logs",
    logging_steps=100,
    save_steps=1000, 
    save_total_limit=2,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to=["none"],
    max_steps = (len(tokenized["train"]) // 16) * 4,  # (데이터샘플수 / 배치사이즈) * 에포크,
    resume_from_checkpoint=False
)

from transformers import EarlyStoppingCallback

# Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]
)

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)  # 옵티마이저 신규 생성
trainer.optimizer = optimizer

#학습률 스케줄러 재설정

from transformers import get_linear_schedule_with_warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=trainer.args.max_steps
)
trainer.lr_scheduler = scheduler

# 학습 시작 + 결과 저장
train_result = trainer.train()
metrics = train_result.metrics
trainer.save_model()
trainer.save_metrics("train", metrics)
trainer.save_state()