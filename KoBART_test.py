from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import torch
import pandas as pd
from tqdm import tqdm
from evaluate import load as load_metric

MODEL_PATH = "./train_4_20250613_0149/checkpoint-7000"  
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


TEST_TSV_PATH = "./datasets/kobart_test.tsv"  
df = pd.read_csv(TEST_TSV_PATH, sep="\t")

predictions = []
for sent in tqdm(df["source"], desc="KoBART 문법 교정 중"):
    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128).to(device)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']

    output_ids = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )
    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    predictions.append(pred)
    
references = df["target"].tolist()


# BLEU
filtered = [(p, r) for p, r in zip(predictions, references) if p and r]
preds, refs = zip(*filtered)
refs = [[ref] for ref in refs]  

metric = load_metric("sacrebleu")
metric.add_batch(predictions=preds, references=refs)
print("BLEU 점수:", metric.compute())


# 저장
df_out = pd.DataFrame({ "source": df["source"], 
                       "target": df["target"], 
                       "prediction": predictions }) 
df_out.to_csv("test_result_20250613_0149.tsv", sep="\t", index=False) 
print("결과 TSV 저장 완료")