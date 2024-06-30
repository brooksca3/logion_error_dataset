import torch
import numpy as np
import re
import unicodedata
import math
from collections import Counter

# Softmax for probability calculation
sm = torch.nn.Softmax(dim=-1)
torch.set_grad_enabled(False)


def deaccent(text):
    # Decompose the string into base characters and combining characters
    decomposed = unicodedata.normalize('NFD', text)

    # Filter out the combining characters (diacritics)
    stripped = ''.join(ch for ch in decomposed if unicodedata.category(ch) != 'Mn')

    return stripped.lower()

with open('YOUR_TRAIN_FILE', 'r') as f:
    full_text = f.read()
full_text = re.sub(r'\s+', ' ', full_text)
full_text = deaccent(full_text)
words = full_text.split()

# Count occurrences of each word
word_counts = Counter(words)

# Create a set of words that appear at least 5 times
full_set = {word for word, count in word_counts.items() if count >= 5}

def compute_token_probabilities2(sentence, model, tokenizer):
    probabilities = []
    # tokens_tensor = torch.tensor([tokenized_sentence]).to(model.device)
    tokens_tensor = torch.tensor(sentence, dtype=torch.long).unsqueeze(0).to(model.device)

    # Predict probabilities for masked token
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
        
    for index in range(1,len(sentence)-1):
        if sentence[index] == tokenizer.mask_token_id:
            continue
        predicted_probabilities = predictions[0, index].softmax(dim=0)
        original_probability = predicted_probabilities[sentence[index]].item()
        probabilities.append(original_probability)

    return probabilities

def generate_combinations(word):
    greek_chars = [
        'ρ', 'χ', 'μ', 'ε', 'ν', 'ο', 'ς', 'σ', 'φ', 'β', 'π', 'α', 'λ', 'ι', 'γ', 'ω', 
        'κ', 'τ', 'δ', 'υ', 'η', 'ζ', 'θ', 'ξ', 'ψ'
    ]
    combinations = set()

    # Adding characters
    for i in range(len(word) + 1):
        for char in greek_chars:
            combinations.add(word[:i] + char + word[i:])

    # Replacing characters
    for i in range(len(word)):
        for char in greek_chars:
            combinations.add(word[:i] + char + word[i+1:])

    # Deleting characters
    for i in range(len(word)):
        combinations.add(word[:i] + word[i+1:])

    return list(combinations)

def sumlog(probabilities):
    return np.sum(np.log(probabilities))

def PLL(sentence, model, tokenizer):
    return sumlog(compute_token_probabilities2(sentence, model, tokenizer))

def get_variations(word):
    combos = generate_combinations(word)
    reduced_combos = []
    for comb in combos:
        valid = True
        for w in comb.split():
            if w not in full_set:
                valid = False
                break
        if valid:
            reduced_combos.append(comb)
    return reduced_combos

def calculate_PLLR(sentence, index, model, tokenizer):
    word = deaccent(sentence.split()[index])
    variations = get_variations(word)
    og_score = PLL(tokenizer.encode(sentence), model, tokenizer)
    chunks = sentence.split()
    champ = og_score
    cur_str = ''
    for var in variations:
        chunks[index] = var
        hypothesis = ' '.join(chunks)
        score = PLL(tokenizer.encode(hypothesis), model, tokenizer)
        if score > champ:
            champ = score
            cur_str = var
    print(f'{word} --> {cur_str}')
    return champ / og_score