from transformers import pipeline

# Create a model for MASK filling
unmasker = pipeline("fill-mask", model="bert-base-uncased")

# Sentence with MASK tokens for words you want to fill in
sentence = "I love my [MASK]."

# Run the model on the sentence to fill in the MASK tokens
result = unmasker(sentence)

# Print the result
print(result)

from transformers import pipeline

# Create a sentiment analysis model
classifier = pipeline("sentiment-analysis")

# Use the classifier to analyze sentiment in the given texts
classifier([
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
    "I love my family",
    "I hate you"
])

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Class to create a GPT-2 model for text generation
class GPT2_Model:
    def __init__(self):
        # Load the GPT-2 tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def generate_text(self, prompt, max_length=50, temperature=0.7):
        # Encode the input prompt and generate text using GPT-2
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(input_ids, max_length=max_length, temperature=temperature)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

# Create an instance of the GPT2_Model class and generate text
gpt2_instance = GPT2_Model()
generated_text = gpt2_instance.generate_text("Your prompt here", max_length=100, temperature=0.9)
print(generated_text)
