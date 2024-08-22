from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Load necessary libraries for data processing and machine learning
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub

# Unzipping the data file
!unzip drive/MyDrive/bioinformatica/Colab_Notebooks/final_project/basic_analayze_data/record_kids_data/data.zip

import zipfile
import os

# Path to the ZIP file
zip_path = 'drive/MyDrive/bioinformatica/Colab_Notebooks/final_project/basic_analayze_data/record_kids_data/data.zip'

# List to store data
data_list = []

# Open the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Iterate through all files in the ZIP
    for file_info in zip_ref.infolist():
        # Check if the file is a text file
        if file_info.filename.endswith('.cha'):
            # Check if the file name contains specific words
            if 'finished' in file_info.filename and 'Braunwald' in file_info.filename:
                match = re.search(r'(\d+)', file_info.filename)
                if match:
                    number = match.group(1)

                # Add data to the list
                data_list.append({'Name:':'Braunwald','Age:': number})

# Create DataFrame from the list
df = pd.DataFrame(data_list)

# Save DataFrame to Excel
excel_output_path = 'drive/MyDrive/Colab_Notebooks/Noa.xlsx'
df.to_excel(excel_output_path, index=False)

print(f'DataFrame saved successfully to Excel file: {excel_output_path}')

# Load additional required libraries
import pandas as pd
import os
import re

# Path to the ZIP file
zip_path = 'drive/MyDrive/Colab_Notebooks/Noa/data.zip'

# List to store data
data_list = []

# Open the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Iterate through all files in the ZIP
    for file_info in zip_ref.infolist():
        # Check if the file name contains specific words
        if 'finished' in file_info.filename and 'Brown/Adam' in file_info.filename:
            # Extract content from the file
            with zip_ref.open(file_info.filename) as file:
                content = file.read().decode('utf-8')

                # Split the content into sentences
                sentences = content.split('\n')
                match = re.search(r'(\d+)', file_info.filename)
                if match:
                    number = match.group(1)

                # Calculate average sentence length
                total_length = sum(len(sentence.strip()) for sentence in sentences)
                average_length = total_length / len(sentences)

                # Add data to the list
                data_list.append({'File': file_info.filename, 'Average Sentence Length_Adam': average_length})

# Create DataFrame from the list
df = pd.DataFrame(data_list)

# Save DataFrame to Excel
df.to_excel(excel_output_path, index=False)

print(f'DataFrame saved successfully to Excel file: {excel_output_path}')

# Repeat similar processes for other names such as Eve, Sarah, Julia, etc., 
# but this time avoid duplicating similar code blocks.

# For example:
# Path to the ZIP file
zip_path = 'drive/MyDrive/Colab_Notebooks/Noa/data.zip'

# List to store data
data_list = []

# Open the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for file_info in zip_ref.infolist():
        if 'finished' in file_info.filename and 'Brown/Eve' in file_info.filename:
            with zip_ref.open(file_info.filename) as file:
                content = file.read().decode('utf-8')
                sentences = content.split('\n')
                match = re.search(r'(\d+)', file_info.filename)
                if match:
                    number = match.group(1)

                total_length = sum(len(sentence.strip()) for sentence in sentences)
                average_length = total_length / len(sentences)

                data_list.append({'File': file_info.filename, 'Average Sentence Length_Eve': average_length})

df = pd.DataFrame(data_list)
df.to_excel(excel_output_path, index=False)

print(f'DataFrame saved successfully to Excel file: {excel_output_path}')

# Similarly, this approach can be used for other data processing steps without duplicating similar blocks of code.
