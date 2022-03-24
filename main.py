import os
import sys
import pandas as pd
_CUR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0,_CUR_PATH)

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, load_metric

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

#### PRE-LOAD ####
# check device
if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
# load fine-tuned model, tokenizer 
tokenizer = AutoTokenizer.from_pretrained('rurupang/roberta-base-finetuned-sts')
model = AutoModelForSequenceClassification.from_pretrained("rurupang/roberta-base-finetuned-sts", num_labels=1)
model.to(device)
##### PRE-LOAD ####

class Data(BaseModel):
    sentence1: str
    sentence2: str

def preprocess_function(examples):
  return tokenizer(examples["sentence1"], examples["sentence2"],
                     truncation=True, max_length=512, padding=True)

def compute_metrics(eval_pred):
  metric_pearsonr = load_metric("pearsonr")
  predictions, labels = eval_pred
  predictions = predictions[:, 0]
  pr = metric_pearsonr.compute(predictions=predictions,
                                  references=labels)
  return pr


@app.post("/")
def inference(request: Data):
  # get_data
  data = request.sentence1
  my_dict = {"sentence1": [request.sentence1], "sentence2": [request.sentence2]}
  df = pd.DataFrame(my_dict)
  dataset = Dataset.from_pandas(df)

  # 토크나이징
  encoded = dataset.map(preprocess_function)

  test_args = TrainingArguments(
      output_dir = _CUR_PATH,
      do_train = False,
      do_predict = True, #eval 단계로 설정
      per_device_eval_batch_size = 1,
      dataloader_drop_last = False    
  )

  # Inference
  trainer = Trainer(
      model = model, 
      args = test_args, 
      compute_metrics = compute_metrics)
  
  test_results = trainer.predict(encoded)
  result = test_results[0][0][0]
  res = 'Yes' if result >= 3 else 'No'
  return {'sentence1': request.sentence1, 'sentence2': request.sentence2, 'Similarity[0-5]':str(result), 'Is it Similar?':res}
