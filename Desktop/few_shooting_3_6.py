from google.colab import drive
drive.mount('/content/drive', force_remount=True)

!ls drive/MyDrive
!find drive/MyDrive
!ls drive/MyDrive/bioinformatica/Colab_Notebooks/final_project/basic_analayze_data/sen_in_csv/sen_3.csv
!unzip drive/MyDrive/bioinformatica/Colab_Notebooks/final_project/basic_analayze_data/sen_in_csv/sen_3.csv

import csv
import random

# Load the CSV data into a list
data = []
with open('drive/MyDrive/bioinformatica/Colab_Notebooks/final_project/basic_analayze_data/sen_in_csv/sen_3.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        sentence, label = row
        cleaned_sentence = sentence.replace('.', '').replace('\n', '')  # Clean sentences by removing periods and newlines
        data.append((cleaned_sentence, label))

# Separate sentences by label (adult/young)
adult_sentences = [s for s, l in data if l == 'adult']
young_sentences = [s for s, l in data if l == 'young']

# Randomly select 20 sentences from each category
random_adult = random.sample(adult_sentences, 20)
random_young = random.sample(young_sentences, 20)

# Display the selected examples
print("Learn the following examples:")
all_examples = [(f"{i}. \"{s}\" label: young", s) for i, s in enumerate(random_young, start=1)]
all_examples += [(f"{i}. \"{s}\" label: adult", s) for i, s in enumerate(random_adult, start=len(all_examples) + 1)]

for i, example in enumerate(all_examples, start=1):
    print(example[0], end=" ")
    if i % 5 == 0:
        print()

# Randomly select 100 sentences for analysis
random_sentences = random.sample(data, 100)

# Print the sentences and their labels
print('Sentences:')
sentences = [sentence for sentence, _ in random_sentences]
print(sentences)

print('\nLabels:')
labels = [label for _, label in random_sentences]
print(labels)
print('\nPredict young or adult by the previous examples: ')

def format_lines(input_text):
    """Function to format lines of text by adding quotes and commas."""
    lines = input_text.split('\n')
    formatted_lines = [f'"{line.strip()}",' for line in lines if line.strip()]
    result = '\n'.join(formatted_lines)
    return result

# Example usage of format_lines function
input_text = """Your example text here..."""
formatted_text = format_lines(input_text)
print(formatted_text)

# Load CSV data for another file (sen_6.csv)
data = []
with open('drive/MyDrive/bioinformatica/Colab_Notebooks/final_project/basic_analayze_data/sen_in_csv/sen_6.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        sentence, label = row
        cleaned_sentence = sentence.replace('.', '').replace('\n', '')  # Clean sentences
        data.append((cleaned_sentence, label))

# Separate sentences by label and randomly select 20 from each
adult_sentences = [s for s, l in data if l == 'adult']
young_sentences = [s for s, l in data if l == 'young']
random_adult = random.sample(adult_sentences, 20)
random_young = random.sample(young_sentences, 20)

# Display examples with labels
print("Learn the following examples:")
all_examples = [(f"{i}. \"{s}\" label: young", s) for i, s in enumerate(random_young, start=1)]
all_examples += [(f"{i}. \"{s}\" label: adult", s) for i, s in enumerate(random_adult, start=len(all_examples) + 1)]

for i, example in enumerate(all_examples, start=1):
    print(example[0], end=" ")
    if i % 5 == 0:
        print()

# Randomly select 100 sentences for analysis
random_sentences = random.sample(data, 100)
print('Sentences:')
sentences = [sentence for sentence, _ in random_sentences]
print(sentences)

print('\nLabels:')
labels = [label for _, label in random_sentences]
print(labels)

# Define the true labels and predicted labels for evaluation
true_labels = ['...']  # Insert actual labels
predicted_labels = ['...']  # Insert predicted labels

# Calculate evaluation metrics
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, pos_label='adult')
recall = recall_score(true_labels, predicted_labels, pos_label='adult')
precision = precision_score(true_labels, predicted_labels, pos_label='adult')

# Display the evaluation results
print(f"\nAccuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
