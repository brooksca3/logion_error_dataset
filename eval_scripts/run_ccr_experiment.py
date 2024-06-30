from ccr_utils import get_ccr_for_word
import ast
import torch
from transformers import BertTokenizer, BertForMaskedLM
from sklearn.metrics import roc_auc_score as roc

print('logion 15 new')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('YOUR PATH HERE')
model = BertForMaskedLM.from_pretrained('YOUR PATH HERE').to(device)

model.eval()

with open('dataset_files/errors1.txt', 'r') as file:
    combined_reports1 = [ast.literal_eval(line.strip()) for line in file]
with open('dataset_files/errors5.txt', 'r') as file:
    combined_reports5 = [ast.literal_eval(line.strip()) for line in file]
with open('dataset_files/true_negatives.txt', 'r') as file:
    true_negatives = [ast.literal_eval(line.strip()) for line in file]

labels = []
ccrs = []
for ind, rep in enumerate(combined_reports1 + combined_reports5 + true_negatives):
    if rep['Label'] == 'GOOD FLAG.':
        labels.append(1)
        ccrs.append(get_ccr_for_word(rep['Text'], rep['Single Index'], model, tokenizer))
        print(f"ind: {ind}, label: {labels[-1]}, ccr: {ccrs[-1]}")
    elif rep['Label'] == 'BAD.':
        labels.append(0)
        ccrs.append(get_ccr_for_word(rep['Text'], rep['Single Index'], model, tokenizer))
        print(f"ind: {ind}, label: {labels[-1]}, ccr: {ccrs[-1]}")
print(labels)
print(ccrs)
print("ROC:", roc(labels, ccrs))