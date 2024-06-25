from transformers import ElectraConfig, ElectraForPreTraining, ElectraForMaskedLM, BertForMaskedLM, ElectraTokenizer, DataCollatorForLanguageModeling
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import getpass
import os

# Configuration
epochs = 100
mask_proportion = 0.15
main_directory = 'main-directory'
batch_size = 16

# Initialize tokenizer
pretrained_path = 'YOUR PRETRAINED PATH HERE'
tokenizer = ElectraTokenizer.from_pretrained(pretrained_path)

train_path = 'YOUR TRAIN SET PATH HERE'
with open(train_path, 'r') as f:
    train = f.readlines()
    if not train[-1].strip():
        train.pop(-1)

# Tokenize training data
train_inputs = tokenizer(train, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
train_inputs['labels'] = train_inputs.input_ids.detach().clone()

# Read validation data
val_path = 'YOUR VAL SET PATH HERE'
with open(val_path, 'r') as f:
    val = f.readlines()
    if not val[-1].strip():
        val.pop(-1)

# Tokenize validation data
val_inputs = tokenizer(val, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
val_inputs['labels'] = val_inputs.input_ids.detach().clone()

# Define file paths and device
username = getpass.getuser()
filestem = os.path.join('/scratch/gpfs/', username, main_directory)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")

# Load pre-trained BERT model for masked language modeling
pretrained_generator_path = 'YOUR PRETRAINED GENERATOR PATH HERE' # we want to use the pretrained embeddings
bert_model = BertForMaskedLM.from_pretrained(pretrained_generator_path).to(device)
bert_embeddings = bert_model.get_input_embeddings()

# Initialize configuration and models
config_generator = ElectraConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=512,
    hidden_size=256,
    num_hidden_layers=12,
    num_attention_heads=4,
    intermediate_size=1024
)

config_discriminator = ElectraConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=512,
    hidden_size=256,
    num_hidden_layers=12,
    num_attention_heads=4,
    intermediate_size=1024,
    embedding_size=bert_embeddings.embedding_dim  # Match pretrained bert embedding size
)

generator = ElectraForMaskedLM(config=config_generator).to(device)
discriminator = ElectraForPreTraining(config=config_discriminator).to(device)

# Copy pretrained embeddings to discriminator
with torch.no_grad():
    discriminator.electra.embeddings.word_embeddings.weight.copy_(bert_embeddings.weight)

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx].clone().detach() for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# Create dataset objects
train_dataset = BaseDataset(train_inputs)
val_dataset = BaseDataset(val_inputs)

# Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mask_proportion)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

# Optimizers
optimizer_gen = optim.AdamW(generator.parameters(), lr=5e-5)
optimizer_disc = optim.AdamW(discriminator.parameters(), lr=5e-5)

# Training loop
for epoch in range(epochs):
    generator.train()
    discriminator.train()
    
    for batch in train_loader:
        inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Generate fake tokens using the generator
        gen_outputs = generator(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        gen_loss = gen_outputs.loss
        fake_logits = gen_outputs.logits
        fake_tokens = inputs.clone()

        # Identify masked positions and sample from softmax distribution
        mask_positions = labels != -100  # -100 is the default value for ignored positions
        softmax_logits = torch.softmax(fake_logits[mask_positions], dim=-1)
        sampled_tokens = torch.multinomial(softmax_logits, num_samples=1).squeeze(-1)
        
        # Replace only the incorrect predictions
        correct_predictions = sampled_tokens == labels[mask_positions]
        fake_tokens[mask_positions] = sampled_tokens

        # Create discriminator labels: 1 for real, 0 for fake at masked positions, but keep real for correct predictions
        discriminator_labels = torch.ones_like(labels).to(device)
        discriminator_labels[mask_positions] = 0
        discriminator_labels[mask_positions][correct_predictions] = 1

        # Train discriminator with mixed tokens (real and fake at masked positions)
        disc_outputs = discriminator(input_ids=fake_tokens, attention_mask=attention_mask, labels=discriminator_labels)

        disc_loss = disc_outputs.loss

        # Backpropagation for generator and discriminator
        optimizer_gen.zero_grad()
        gen_loss.backward()
        optimizer_gen.step()

        optimizer_disc.zero_grad()
        disc_loss.backward()
        optimizer_disc.step()
        
    print(f'Epoch {epoch+1}/{epochs}, Generator Loss: {gen_loss.item()}, Discriminator Loss: {disc_loss.item()}')

    # Evaluate on validation set
    generator.eval()
    discriminator.eval()
    val_gen_loss = 0
    val_disc_loss = 0
    
    with torch.no_grad():
        do_first = True
        for batch in val_loader:
            inputs, labels = batch['input_ids'].to(device), batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Generate fake tokens
            gen_outputs = generator(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            val_gen_loss += gen_outputs.loss.item()
            fake_logits = gen_outputs.logits
            fake_tokens = inputs.clone()
            mask_positions = labels != -100
            softmax_logits = torch.softmax(fake_logits[mask_positions], dim=-1)
            sampled_tokens = torch.multinomial(softmax_logits, num_samples=1).squeeze(-1)
            
            # Replace only the incorrect predictions
            correct_predictions = sampled_tokens == labels[mask_positions]
            fake_tokens[mask_positions] = sampled_tokens

            # Ensure no extra tokens are inserted
            assert fake_tokens.size() == inputs.size(), "Size mismatch after replacing tokens"

            # Create discriminator labels for validation
            discriminator_labels = torch.ones_like(labels).to(device)
            discriminator_labels[mask_positions] = 0
            discriminator_labels[mask_positions][correct_predictions] = 1

            # Evaluate discriminator with mixed tokens
            disc_outputs = discriminator(input_ids=fake_tokens, attention_mask=attention_mask, labels=discriminator_labels)
            
            val_disc_loss += disc_outputs.loss.item()
    
    print(f'Validation Generator Loss: {val_gen_loss / len(val_loader)}, Validation Discriminator Loss: {val_disc_loss / len(val_loader)}')

    # Save the model after each epoch
    # generator.save_pretrained(os.path.join(filestem, 'content_electra_generator', str(batch_size), f'generator_epoch_{epoch+1}'))
    discriminator.save_pretrained(os.path.join(filestem, 'content_electra_discriminator_gen', str(batch_size), f'discriminator_epoch_{epoch+1}'))

# Final save after training
# generator.save_pretrained(os.path.join(filestem, 'content_electra_generator', str(batch_size), 'generator_final'))
discriminator.save_pretrained(os.path.join(filestem, 'content_electra_discriminator_gen', str(batch_size), 'discriminator_final'))