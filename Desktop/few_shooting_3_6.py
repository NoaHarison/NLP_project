import csv
import random
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

# Load and clean the data from CSV file
data = []
with open('drive/MyDrive/bioinformatica/Colab_Notebooks/final_project/basic_analayze_data/sen_in_csv/sen_3.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        sentence, label = row
        cleaned_sentence = sentence.replace('.', '').replace('\n', '')  # Remove periods and newlines
        data.append((cleaned_sentence, label))

# Separate adult and young sentences
adult_sentences = [s for s, l in data if l == 'adult']
young_sentences = [s for s, l in data if l == 'young']

# Task 1: Randomly select 20 sentences from each category and print them with labels
random_adult = random.sample(adult_sentences, 20)
random_young = random.sample(young_sentences, 20)

print("Learn the following examples:")
all_examples = [(f"{i}. \"{s}\" label: young", s) for i, s in enumerate(random_young, start=1)]
all_examples += [(f"{i}. \"{s}\" label: adult", s) for i, s in enumerate(random_adult, start=len(all_examples)+1)]

for i, example in enumerate(all_examples, start=1):
    print(example[0], end=" ")
    if i % 5 == 0:
        print()

# Task 2: Randomly select 100 sentences and print them as a list
random_sentences = random.sample(data, 100)

print('\nSentences:')
sentences = [sentence for sentence, _ in random_sentences]
print(sentences)

print('\nLabels:')
labels = [label for _, label in random_sentences]
print(labels)

print('\npredict young or adult by the previous examples: ')

# Function to format text with quotes and commas
def format_lines(input_text):
    lines = input_text.split('\n')
    formatted_lines = [f'"{line.strip()}",' for line in lines if line.strip()]
    result = '\n'.join(formatted_lines)
    return result

# Example usage of the format_lines function
input_text = """down up an l a ten young
mhm it's way down to the ground up adult
he hide kitty hiding yeah young"""
formatted_text = format_lines(input_text)
print(formatted_text)

# Task 3: Tag sentences based on the presence of 'young' or 'adult'
sentences = ["down up an l a ten young", "mhm it's way down to the ground up adult"]
tags = ['young' if 'young' in line else 'adult' for line in sentences]
print(tags)

# Evaluate predicted vs. true labels
true_labels = ['adult', 'young', 'young', 'young', 'adult']
predicted_labels = ['young', 'adult', 'young', 'young', 'young']

accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels, pos_label='adult')
recall = recall_score(true_labels, predicted_labels, pos_label='adult')
precision = precision_score(true_labels, predicted_labels, pos_label='adult')

print(f"\nAccuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")

# Load and clean another CSV file (sen_6.csv) and repeat similar tasks
data = []
with open('drive/MyDrive/bioinformatica/Colab_Notebooks/final_project/basic_analayze_data/sen_in_csv/sen_6.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        sentence, label = row
        cleaned_sentence = sentence.replace('.', '').replace('\n', '')  # Remove periods and newlines
        data.append((cleaned_sentence, label))

# Separate adult and young sentences
adult_sentences = [s for s, l in data if l == 'adult']
young_sentences = [s for s, l in data if l == 'young']

# Randomly select 20 from each category
random_adult = random.sample(adult_sentences, 20)
random_young = random.sample(young_sentences, 20)

print("Learn the following examples:")
all_examples = [(f"{i}. \"{s}\" label: young", s) for i, s in enumerate(random_young, start=1)]
all_examples += [(f"{i}. \"{s}\" label: adult", s) for i, s in enumerate(random_adult, start=len(all_examples)+1)]

for i, example in enumerate(all_examples, start=1):
    print(example[0], end=" ")
    if i % 5 == 0:
        print()

# Randomly select 100 sentences from the sen_6.csv
random_sentences = random.sample(data, 100)

print('\nSentences:')
sentences = [sentence for sentence, _ in random_sentences]
print(sentences)

print('\nLabels:')
labels = [label for _, label in random_sentences]
print(labels)
