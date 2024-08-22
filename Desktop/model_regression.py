# -*- coding: utf-8 -*-
"""model_Regrresion.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kVRAH17X_3bjA9YXigzMcJ6jNp8zKzrf
"""

from google.colab import drive  # connect to drive
drive.mount('/content/drive')

!pip install transformers datasets  # install required packages
!pip install accelerate -U

# After these installations, you need to press RunTime --> Interrupt Session!

import pandas as pd
from datasets import Dataset
from datasets import load_dataset

# Loading the data
df = pd.read_csv('drive/MyDrive/bioinformatica/Colab_Notebooks/final_project/basic_analayze_data/csv_for_hugging_face/sen_6.csv', skiprows=1, names=['Text', 'Age'])
df['Text'] = df['Text'].str.replace('\r', '')  # remove carriage returns
df['Age'] = df['Age'].astype(float)  # convert Age column to float

# Converts the data from Pandas to Hugging Face so we can use it with transformers and natural language processing models
mydataset = Dataset.from_pandas(df)
mydataset.to_pandas()  # So that we can present in a convenient way
print(mydataset)

# Library imports
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding, AutoTokenizer  # import AutoTokenizer
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import AutoModelForSequenceClassification

# Convert dataset to DataFrame and then to text lists and labels
texts = mydataset.to_pandas()["Text"].tolist()
labels = [float(i) for i in mydataset.to_pandas()["Age"].tolist()]

# Divide the data into 3 groups: training, validation, and test (80% training, 10% validation, 10% test)
train_texts, tmp_texts, train_labels, tmp_labels = train_test_split(texts, labels, test_size=0.2)
val_texts, test_texts, val_labels, test_labels = train_test_split(tmp_texts, tmp_labels, test_size=0.5)

# Load the BERT tokenizer, which will be used to tokenize the texts
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Perform tokenization of the texts in each of the groups (training, validation, test), truncating and padding to adjust texts to a fixed length
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Define a class that creates a PyTorch dataset from the token and label arrays
class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)  # convert labels to float
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset objects for training, validation, and test sets
train_dataset = TextClassificationDataset(train_encodings, train_labels)
val_dataset = TextClassificationDataset(val_encodings, val_labels)
test_dataset = TextClassificationDataset(test_encodings, test_labels)

# Data collator, divide to batch, do padding, pack the data after padding - technical details
data_collator = DataCollatorWithPadding(tokenizer)

# This line imports the AutoModelForSequenceClassification model from Hugging Face's Transformers library
from transformers import AutoModelForSequenceClassification

# This line loads the base BERT model and fits it to a regression problem (predicting one numerical value)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", problem_type="regression", num_labels=1)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="drive/MyDrive/data_sets/Bert_Cola/test_trainer/",  # output directory
    overwrite_output_dir=True,
    learning_rate=2e-5,  # learning rate for optimizing weights during training
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=16,  # batch size for evaluation
    num_train_epochs=15,  # number of epochs to train the model
    weight_decay=0.01,  # normalizing the weights
    eval_strategy="no",  # evaluation strategy: no evaluation during training
    save_strategy="no",  # save strategy: no saving during training
    load_best_model_at_end=True,  # load the best model at the end of training
    push_to_hub=False,  # don't push the model to the hub
    logging_steps=20  # logging steps
)

import numpy as np
from datasets import load_metric
import torch.nn.functional as F

# Load the MSE metric
metric = load_metric("mse")

# Compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.squeeze()
    mse = ((predictions - labels) ** 2).mean().item()
    return {"mse": mse}

import torch.nn.functional as F
from transformers import Trainer
import torch
from transformers import Trainer

# Custom Trainer class for computing loss using MSE
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss_fn = F.mse_loss
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Instantiate the Trainer class
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Save the trained model
model.save_pretrained("drive/MyDrive")

# Run the trained model on the validation set and compute metrics
val_predictions = trainer.predict(val_dataset)
val_preds = val_predictions.predictions.squeeze()

# Compute regression metrics on the validation set
val_true = val_dataset.labels
val_mse = mean_squared_error(val_true, val_preds)  # MSE: average squared error
val_rmse = np.sqrt(val_mse)  # RMSE: root of MSE, similar to standard deviation
val_mae = mean_absolute_error(val_true, val_preds)  # MAE: average absolute error
val_r2 = r2_score(val_true, val_preds)  # R²: coefficient of determination

print(f"Validation Mean Squared Error (MSE): {val_mse:.4f}")
print(f"Validation Root Mean Squared Error (RMSE): {val_rmse:.4f}")
print(f"Validation Mean Absolute Error (MAE): {val_mae:.4f}")
print(f"Validation R² (Coefficient of Determination): {val_r2:.4f}")

# Test the model on the test set and compute metrics
test_predictions = trainer.predict(test_dataset)
test_preds = test_predictions.predictions.squeeze()

# Compute regression metrics on the test set
test_true = test_dataset.labels
test_mse = mean_squared_error(test_true, test_preds)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(test_true, test_preds)
test_r2 = r2_score(test_true, test_preds)

print(f"Test Mean Squared Error (MSE): {test_mse:.4f}")
print(f"Test Root Mean Squared Error (RMSE): {test_rmse:.4f}")
print(f"Test Mean Absolute Error (MAE): {test_mae:.4f}")
print(f"Test R² (Coefficient of Determination): {test_r2:.4f}")
