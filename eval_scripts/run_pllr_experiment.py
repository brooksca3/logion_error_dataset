import ast
import torch
import json
from transformers import BertTokenizer, BertForMaskedLM
from pllr_utils import calculate_PLLR
from sklearn.metrics import roc_auc_score as roc


print('doing 40')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('YOUR PATH HERE')
model = BertForMaskedLM.from_pretrained('YOUR PATH HERE').to(device)

model.eval()

with open('dataset_files/errors_split_1.json', 'r') as file:
    combined_reports1 = json.load(file)
with open('dataset_files/errors_split_5.json', 'r') as file:
    combined_reports5 = json.load(file)
with open('dataset_files/random_assumed_true_negatives.json', 'r') as file:
    true_negatives = json.load(file)

labels = []
ccrs = []
for ind, rep in enumerate(combined_reports1 + combined_reports5 + true_negatives):
    if rep['Label'] == 'GOOD FLAG.':
        labels.append(1)
        ccrs.append(calculate_PLLR(rep['Text'], rep['Single Index'], model, tokenizer))
        print(f"ind: {ind}, label: {labels[-1]}, ccr: {ccrs[-1]}")
    elif rep['Label'] == 'BAD.':
        labels.append(0)
        ccrs.append(calculate_PLLR(rep['Text'], rep['Single Index'], model, tokenizer))
        print(f"ind: {ind}, label: {labels[-1]}, ccr: {ccrs[-1]}")
print(labels)
print(ccrs)
print("ROC:", roc(labels, ccrs))