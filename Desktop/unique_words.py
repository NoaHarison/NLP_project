import re
from nltk.corpus import words
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive

# Mount Google Drive to access files
drive.mount('/content/drive', force_remount=True)

# Load a set of English words from the nltk corpus
english_words = set(words.words())

# Function to check if a word is English
def is_english_word(word):
    # Remove non-alphabetic characters from the word
    word = re.sub(r'[^a-zA-Z]', '', word)
    return word.lower() in english_words

# Path to the ZIP file
zip_path = 'drive/MyDrive/bioinformatica/Colab_Notebooks/final_project/basic_analayze_data/record_kids_data/data.zip'
# List to store file contents and number of unique English words
file_contents = []

# Open the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Loop through each file in the ZIP
    for file_info in zip_ref.infolist():
        # Check if the file is a text file containing 'finished' and 'Brown/Sarah'
        if 'finished' in file_info.filename and 'Brown/Sarah' in file_info.filename:
            # Read the file contents
            with zip_ref.open(file_info.filename) as file:
                contents = file.read().decode('utf-8')
            # Split the contents into sentences (each line is a sentence)
            sentences = contents.split('\\n')
            # Get the number sequence from the file name
            file_nums = ''.join(filter(str.isdigit, file_info.filename))
            # Filter out non-English words from each sentence
            english_words_list = []
            for sentence in sentences:
                words = re.findall(r'\w+(?:-\w+)*', sentence)
                english_words_list.extend([word for word in words if is_english_word(word)])
            # Count unique English words
            unique_english_words = len(set(english_words_list))
            # Add the number of unique English words to the list
            file_contents.append((file_nums, unique_english_words))

# Create a new Excel file
workbook = Workbook()
worksheet = workbook.active
# Write column headers
worksheet['A1'] = 'Number Sequence'
worksheet['B1'] = 'Unique English Words'
# Write data to the worksheet
for row, (file_nums, unique_english_words) in enumerate(file_contents, start=2):
    worksheet.cell(row=row, column=1, value=file_nums)
    worksheet.cell(row=row, column=2, value=unique_english_words)

# Save the Excel file
excel_file_path = 'drive/MyDrive/bioinformatica/Colab_Notebooks/final_project/basic_analayze_data.xlsx'
workbook.save(excel_file_path)
print(f"Excel file saved successfully: {excel_file_path}")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data from the Excel file
data_df = pd.read_excel('drive/MyDrive/bioinformatica/Colab_Notebooks/final_project/basic_analayze_data/different_word_in_english/TRY_SARAH.xlsx')

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(data_df['age'], data_df['Unique English Words'], color='blue', alpha=0.5)

# Fit a quadratic trend line
coeffs = np.polyfit(data_df['age'], data_df['Unique English Words'], 2)
poly = np.poly1d(coeffs)
x_range = np.linspace(data_df['age'].min(), data_df['age'].max(), 100)
y_range = poly(x_range)

# Plot the trend line
plt.plot(x_range, y_range, color='red')

plt.title('Sarah: Unique English Words vs. Age')
plt.xlabel('Age')
plt.ylabel('Unique English Words')
plt.grid(True)
plt.show()
