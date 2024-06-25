# taken and adapted from https://github.com/jamescalam/transformers/blob/main/course/training/03_mlm_training.ipynb
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BertConfig
import torch
import getpass
import sys
epochs = 250
#training_portion = 0.9      # Reserve 10% of dataset for testing
#random_seed = 42            # This seed is used so that we can recover the exact train/test split after this script terminates.
num_logs_per_epoch = 4            # (in steps)
num_evals_per_epoch = 4          # (in steps)
num_saves_per_epoch = 4
mask_proportion = 0.4 # change this mask proportion as desired

main_directory = 'main-directory'

if len(sys.argv) == 1:
        print("Received no argument for batch size. Defaulting to 16.")
        batch_size = 16
elif len(sys.argv) > 1:
        print(f"Setting batch size to {sys.argv[1]}.")
        batch_size = int(sys.argv[1])

pretrained_path = 'YOUR PRETRAINED PATH HERE'
tokenizer = BertTokenizer.from_pretrained(pretrained_path)

train_path = 'YOUR TRAIN SET PATH HERE'
with open(train_path, 'r') as f:
  train = f.readlines()
  if not train[-1].strip():
      train.pop(-1)
  f.close()

train_inputs = tokenizer(train, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
train_inputs['labels'] = train_inputs.input_ids.detach().clone()

val_path = 'YOUR VAL SET PATH HERE'
with open(val_path, 'r') as f:
  val = f.readlines()
  if not val[-1].strip():
      val.pop(-1)
  f.close()

val_inputs = tokenizer(val, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
val_inputs['labels'] = val_inputs.input_ids.detach().clone()

num_steps = len(train) // batch_size
log_every = int(num_steps/num_logs_per_epoch)
eval_every = int(num_steps/num_evals_per_epoch)
save_every = int(num_steps/num_saves_per_epoch)

username = getpass.getuser()
filestem = '/scratch/gpfs/' + username + '/' + main_directory

print(torch.version.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")

config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=512
)

# Initialize model with the configuration and move it to the device
model = BertForMaskedLM(config=config).to(device)

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = BaseDataset(train_inputs)
val_dataset = BaseDataset(val_inputs)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mask_proportion)

training_args = TrainingArguments(
    evaluation_strategy = "steps",
    eval_steps=eval_every,
    logging_steps=log_every,
    save_steps=save_every,
    output_dir=filestem + '/mask40' + str(batch_size),
    per_device_train_batch_size=batch_size,
    num_train_epochs=epochs
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained(filestem + '/content_mask40' + str(batch_size) + '/tester')