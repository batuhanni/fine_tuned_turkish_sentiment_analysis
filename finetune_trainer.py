#%%
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
#%%
df = pd.read_csv(f'dataset.csv')
df.label = df.label.apply(lambda x: 1 if x == "positive" else 0)
df = df[:29760].sample(frac=1).reset_index(drop=True)

df_train = df[:50]
df_test = df[50:60]

dataset = DatasetDict({
    "train": Dataset.from_pandas(df_train),
    "test": Dataset.from_pandas(df_test)
    })

tokenizer = AutoTokenizer.from_pretrained('savasy/bert-base-turkish-sentiment-cased')

def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

dataset = dataset.map(preprocess_function, batched=True)

#%%
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("savasy/bert-base-turkish-sentiment-cased", num_labels=2)

#%%
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
#trainer.save_model()
# %%
