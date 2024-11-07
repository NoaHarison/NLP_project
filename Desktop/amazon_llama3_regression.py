# -*- coding: utf-8 -*-
"""Fine-Tuning LLAMA3 for Sequence Regression on Electronics Dataset"""

# Required installations. Run once in the Linux terminal before running the script:
# pip install "torch==2.4" tensorboard transformers datasets accelerate evaluate bitsandbytes huggingface_hub trl peft matplotlib

import os
import gzip
import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# Step 1: Load the compressed dataset and filter reviews
file_path = '/workspace/Levona/Noa/amazon_Bert_microsoft/Electronics.jsonl.gz'
texts = []
ratings = []
example_printed = False

def is_valid_review(text):
    sentences = text.split('.')
    long_sentences = [s for s in sentences if len(s.split()) >= 15]
    return len(long_sentences) >= 3

with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    for line in f:
        json_obj = json.loads(line.strip())
        if 'text' in json_obj and 'rating' in json_obj and is_valid_review(json_obj['text'].lower()):
            texts.append(json_obj['text'].lower())
            ratings.append(json_obj['rating'])
            if not example_printed:
                print("Example review that meets the criteria:")
                print(json_obj['text'])
                example_printed = True

# Creating a DataFrame from the reviews and ratings
df = pd.DataFrame({
    'text': texts,
    'rating': ratings
})

# Select a sample of 10,000 samples per rating
samples_per_rating = 10000
sampled_df = pd.DataFrame()
for rating in range(1, 6):
    rating_df = df[df['rating'] == float(rating)]
    sampled_rating_df = rating_df.sample(n=samples_per_rating, random_state=42)
    sampled_df = pd.concat([sampled_df, sampled_rating_df])

sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 2: Split into training, validation, and test sets
train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    sampled_df['text'].tolist(),
    sampled_df['rating'].tolist(),
    test_size=0.1,
    random_state=42
)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_val_texts,
    train_val_labels,
    test_size=0.1111,
    random_state=42
)

# Step 3: Tokenization with pad_token setting
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Add pad_token if it does not exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Step 4: Define a custom dataset class
class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextClassificationDataset(train_encodings, train_labels)
val_dataset = TextClassificationDataset(val_encodings, val_labels)
test_dataset = TextClassificationDataset(test_encodings, test_labels)

# Step 5: Define data collator
data_collator = DataCollatorWithPadding(tokenizer)

# Step 6: Define the LLAMA model with Quantized LoRA
model_name = "meta-llama/Llama-3.2-1B"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='SEQ_CLS'
)

# Load the model with QLoRA
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    problem_type="regression",
    num_labels=1
)

# Adjust the model to the new number of tokens
model.resize_token_embeddings(len(tokenizer))

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Step 7: Define training parameters (changing batch size to 1)
training_args = TrainingArguments(
    output_dir="/workspace/Levona/Noa/model_output",
    overwrite_output_dir=True,
    learning_rate=1e-4,
    per_device_train_batch_size=1,  # Changed to 1
    per_device_eval_batch_size=1,   # Changed to 1
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    load_best_model_at_end=True,
    logging_steps=20,
    fp16=True,
    gradient_accumulation_steps=8,
)

# Step 8: Define a custom metric function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions.squeeze())
    return {'rmse': np.sqrt(mse)}

# Step 9: Define a custom Trainer
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Step 10: Run training
trainer.train()

# Step 11: Save the model and evaluate it
trainer.save_model("/workspace/Levona/Noa/model_output")

# Evaluate on the test set
test_predictions = trainer.predict(test_dataset)
test_preds = test_predictions.predictions.squeeze()

# Calculate metrics for the test set
test_true = test_labels
test_mse = mean_squared_error(test_true, test_preds)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(test_true, test_preds)
test_r2 = r2_score(test_true, test_preds)

print(f"Test MSE: {test_mse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test R²: {test_r2:.4f}")

# Step 12: Create a graph of actual vs predicted values
plt.figure(figsize=(8, 8))

# Sample 20% of the data for example purposes
sample_size = int(0.2 * len(test_true))
indices = np.random.choice(range(len(test_true)), size=sample_size, replace=False)

# Scatter plot of true vs predicted values
plt.scatter(test_preds[indices], np.array(test_true)[indices], alpha=0.5)

# Add y=x reference line
plt.plot([1, 5], [1, 5], color='red', linestyle='--')

plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.title('Predicted vs Actual Ratings')
plt.grid(True)
plt.savefig('/workspace/Levona/Noa/model_output/true_vs_predicted.png')
plt.show()
