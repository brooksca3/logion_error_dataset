import torch
import time
from transformers import ElectraTokenizer, ElectraForPreTraining
import json


valid_letters =  '#ςερτυθιοπλκξηγφδσαζχψωβνμ'

def calculate_error_prob(sentence, position, model, tokenizer):
    char_token_ids = tokenizer(sentence, max_length=512, truncation=True)['input_ids']
    original_token_id = char_token_ids[position]  # Save the ground truth token id
    print(f"original: {tokenizer.convert_ids_to_tokens([original_token_id])}")

    print('position: ', position)
    # Mask the current token
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    # Get model output
    with torch.no_grad():
        discriminator_outputs = model(input_ids)
        torch.cuda.empty_cache()

    # Calculate probabilities
    probabilities = discriminator_outputs[0].squeeze()
    # print(probabilities)
    return probabilities[position]

def get_indices(index, string_ls, tokenizer):
    input_ids = tokenizer(string_ls)['input_ids']
    sum = 0
    for i in range(index):
        sum += (len(input_ids[i]) - 2)
    return list(range(sum + 1, sum + len(input_ids[index]) - 1)) ## plus one to avoid CLS

def get_prob_for_word(sentence, word_index, model, tokenizer):
    inds = get_indices(word_index, sentence.split(), tokenizer)
    # print(inds)
    probs = [calculate_error_prob(sentence, ind, model, tokenizer) for ind in inds]
    # print('probs: ', probs)
    return min(probs)

def main():
    
    sentence = "Συνδυάζει ὁ τούτῳ καὶ τὸ παράδειγμα · τὴν γὰρ ἰδέαν ὁ Πλάτων παραδειγματικὸν αἴτιον λέγει ."
    tokenizer = ElectraTokenizer.from_pretrained("YOUR PATH HERE")
    model = ElectraForPreTraining.from_pretrained("YOUR PATH HERE")
    model.eval()
    ## example usage
    print(get_prob_for_word(sentence, 1, model, tokenizer))