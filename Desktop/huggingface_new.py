from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Import necessary libraries
import numpy as np  # For numerical computations
import pandas as pd  # For data manipulation
import re  # For regular expressions and text processing
import tensorflow as tf  # For building and training neural networks
import tensorflow_hub as hub  # For loading pre-trained models from TensorFlow Hub
from transformers import Trainer  # For model training using Hugging Face's Transformers

# Install necessary packages
!pip install transformers
!pip install datasets
!pip install nltk
!pip install transformers[torch]
!pip install accelerate -U
!pip install transformers[tensorflow] -U
!pip install accelerate[tensorflow] -U
!pip install transformers accelerate
!pip install accelerate>=0.20.1

# Load dataset
df = pd.read_csv('drive/MyDrive/bioinformatica/Colab_Notebooks/final_project/spam_or_not/spam_ham_dataset.csv', names=['num', 'label', 'text', 'label_num'])

# Shuffle the dataset and convert label_num to integers
df = shuffle(df)
df['label_num'] = pd.to_numeric(df['label_num'], errors='coerce')
df['label_num'] = df['label_num'].fillna(0).astype(int)

# Split the dataset into training, validation, and test sets
train, test = train_test_split(df[['label_num', 'text']], test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

# Convert pandas DataFrames to Hugging Face Datasets
ds_dict = {
    'train': Dataset.from_pandas(train),
    'val': Dataset.from_pandas(val),
    'test': Dataset.from_pandas(test)
}
ds = DatasetDict(ds_dict)
ds = ds.remove_columns("__index_level_0__")

# Load pre-trained BERT model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize the datasets
tokenized_datasets = ds.map(lambda examples: tokenizer(examples['text'], truncation=True), batched=True)

# Prepare data for TensorFlow training
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

# Convert datasets to TensorFlow datasets
tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "token_type_ids", "label_num"],
    shuffle=True,
    batch_size=4,
    collate_fn=data_collator
)

tf_validation_dataset = tokenized_datasets["val"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "label_num"],
    shuffle=False,
    batch_size=4,
    collate_fn=data_collator
)

# Set up training arguments
training_args = TFTrainingArguments(
    output_dir='',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    evaluation_strategy='epoch'
)

# Initialize the trainer
trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=tf_train_dataset,
    eval_dataset=tf_validation_dataset
)

# Train and evaluate the model
trainer.train()
results = trainer.evaluate()
