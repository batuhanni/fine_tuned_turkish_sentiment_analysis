from django.shortcuts import render
import torch
from torch.utils.data import DataLoader
import pandas as pd
from transformers import AutoTokenizer, get_scheduler
from datasets import Dataset, DatasetDict
from django.core.cache import cache
from torch.optim import AdamW


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = torch.load('sa_api\\turkish_finetuned', map_location=torch.device('cpu'))

def load_model():
    global model
    if model is None:
        model = torch.load('sa_api\\turkish_finetuned', map_location=torch.device('cpu'))
    return model

def train_model(model_for_predict, sentence, label, threelabels=False):
    model_for_predict.to(device)

    tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-sentiment-cased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    print(label)
    label = 1 if label == 'Positive' else 0
    dataset = DatasetDict({
        "train": Dataset.from_pandas(pd.DataFrame(columns=['text','labels'], data=[[sentence, label]]))
        })

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=1)

    optimizer = AdamW(model_for_predict.parameters(), lr=0.000025)

    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model_for_predict.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model_for_predict(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    print("Model has been trained")
    return model_for_predict

def predict(model_for_predict, sentence):
    model_for_predict.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-sentiment-cased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    dataset = DatasetDict({
        "test": Dataset.from_pandas(pd.DataFrame(columns=['text','labels'], data=[[sentence, 0]]))
        })

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets.set_format("torch")

    eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=1)

    model_for_predict.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model_for_predict(**batch)

        logits = outputs.logits
 
        predictions = torch.argmax(logits, dim=-1)
    return predictions.item()

#cache.set('sentence', 'Write your sentence here!', 3600)

def index(request):
    model = load_model()
    if request.method == 'POST':
        if request.POST.get('sentence') != None:
            sentence = request.POST.get('sentence')
            cache.set('sentence', sentence, 3600)
            result = predict(model, sentence)
            result = 'Positive' if result == 1 else 'Negative'
            cache.set('result', result, 3600)
        else:
            sentence = cache.get('sentence')
            result = cache.get('result')
        if (request.POST.get('feedback') != None) and (result != request.POST.get('feedback')):
            model = train_model(model, sentence, request.POST.get('feedback'), threelabels=3)
            feedback_response = 'Model has been fine-tuned with your feedback.'
            torch.save(model, 'sa_api\\turkish_finetuned')
        else:
            feedback_response = 'Model has not been feedbacked.'
    else:
        result = 'Waiting for your sentence...'
        feedback_response = 'Model has not been feedbacked.'

    context = {"result": result,
               "feedback_response": feedback_response}
    
    return render(request, "index.html", context)