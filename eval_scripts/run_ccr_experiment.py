from utils.ccr_utils import get_ccr_for_word
import ast
import json
import torch
from transformers import BertTokenizer, BertForMaskedLM
from sklearn.metrics import roc_auc_score as roc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = BertTokenizer.from_pretrained('YOUR PATH HERE')
model = BertForMaskedLM.from_pretrained('YOUR PATH HERE').to(device)

model.eval()

with open('../dataset_files/errors_split_1.json', 'r') as file:
    combined_reports1 = json.load(file)
with open('../dataset_files/errors_split_5.json', 'r') as file:
    combined_reports5 = json.load(file)
with open('../dataset_files/random_assumed_true_negatives.json', 'r') as file:
    true_negatives = json.load(file)

labels = []
ccrs = []
for ind, rep in enumerate(combined_reports1 + combined_reports5 + true_negatives):
    if rep['Label'] == 'GOOD FLAG.':
        labels.append(1)
        ccrs.append(get_ccr_for_word(rep['Text'], rep['Word Index in Text'], model, tokenizer))
        print(f"ind: {ind}, label: {labels[-1]}, ccr: {ccrs[-1]}")
    elif rep['Label'] == 'BAD.':
        labels.append(0)
        ccrs.append(get_ccr_for_word(rep['Text'], rep['Word Index in Text'], model, tokenizer))
        print(f"ind: {ind}, label: {labels[-1]}, ccr: {ccrs[-1]}")
print(labels)
print(ccrs)
# If your AUROC is less than 0.5, then just do 1 - labels :) For some metrics a low score indicates an error, for other metrics it's a high score
print("ROC:", roc(labels, ccrs))

## Note that there are 763 examples which are either 'GOOD FLAG.' or 'BAD.'
## We used GPT-4 to take in the expert's notes and tag each of these 763 as scribal, print, digital, or bad. 
## Here are those tags, in order (data split 1 and then immediately data split 5):
## notes = ['print', 'digital', 'print', 'digital', 'print', 'digital', 'print', 'bad', 'digital', 'digital', 'print', 'scribal', 'digital', 'digital', 'print', 'print', 'digital', 'bad', 'print', 'digital', 'print', 'scribal', 'print', 'print', 'print', 'print', 'bad', 'digital', 'bad', 'print', 'print', 'print', 'digital', 'print', 'print', 'digital', 'print', 'print', 'scribal', 'bad', 'bad', 'print', 'bad', 'print', 'print', 'print', 'print', 'bad', 'bad', 'scribal', 'scribal', 'bad', 'bad', 'print', 'bad', 'print', 'digital', 'print', 'print', 'scribal', 'scribal', 'print', 'bad', 'bad', 'digital', 'bad', 'bad', 'scribal', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'digital', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'digital', 'print', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'scribal', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'print', 'bad', 'bad', 'bad', 'scribal', 'digital', 'bad', 'bad', 'bad', 'scribal', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'digital', 'bad', 'bad', 'print', 'digital', 'scribal', 'bad', 'print', 'bad', 'digital', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'scribal', 'bad', 'bad', 'bad', 'scribal', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'digital', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'digital', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'print', 'bad', 'bad', 'digital', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'scribal', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'scribal', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'bad', 'bad', 'scribal', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'scribal', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'scribal', 'bad', 'bad', 'bad', 'bad', 'bad', 'scribal', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'scribal', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'digital', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'scribal', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'scribal', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'print', 'print', 'print', 'print', 'print', 'bad', 'digital', 'print', 'print', 'bad', 'digital', 'print', 'print', 'digital', 'bad', 'scribal', 'digital', 'print', 'print', 'bad', 'scribal', 'digital', 'print', 'print', 'bad', 'bad', 'digital', 'bad', 'bad', 'print', 'digital', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'digital', 'print', 'bad', 'bad', 'print', 'print', 'print', 'digital', 'bad', 'bad', 'bad', 'scribal', 'print', 'scribal', 'print', 'print', 'bad', 'scribal', 'print', 'digital', 'bad', 'bad', 'bad', 'print', 'bad', 'print', 'bad', 'bad', 'bad', 'scribal', 'bad', 'print', 'bad', 'print', 'bad', 'digital', 'scribal', 'print', 'bad', 'print', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'print', 'digital', 'bad', 'print', 'digital', 'scribal', 'bad', 'digital', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'scribal', 'print', 'bad', 'print', 'bad', 'bad', 'scribal', 'scribal', 'bad', 'print', 'bad', 'bad', 'print', 'bad', 'scribal', 'bad', 'bad', 'scribal', 'bad', 'digital', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'scribal', 'bad', 'bad', 'print', 'bad', 'scribal', 'bad', 'bad', 'digital', 'bad', 'bad', 'print', 'bad', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'scribal', 'bad', 'bad', 'bad', 'scribal', 'bad', 'bad', 'bad', 'scribal', 'bad', 'scribal', 'bad', 'bad', 'scribal', 'scribal', 'scribal', 'bad', 'bad', 'scribal', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'scribal', 'bad', 'bad', 'bad', 'bad', 'scribal', 'digital', 'bad', 'bad', 'bad', 'scribal', 'print', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'scribal', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'scribal', 'scribal', 'print', 'print', 'bad', 'bad', 'bad', 'print', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'bad', 'scribal', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'digital', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'scribal', 'scribal', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'scribal', 'bad', 'bad', 'bad', 'bad', 'print', 'scribal', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'scribal', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'scribal', 'print', 'scribal', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'print', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'scribal', 'bad']
