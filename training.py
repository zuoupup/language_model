import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import *


# Setting device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
VOCAB_SIZE = 10000
EMBEDDING_DIM = 512
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 1024
BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 0.00001
MAX_SEQ_LEN = 15  # Increased to accommodate longer sentences



# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, sentences, vocab, max_seq_len):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.sentences = [self.encode_sentence(sentence) for sentence in sentences]

    def encode_sentence(self, sentence):
        tokens = tokenizer(sentence.strip())
        token_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        # Truncate or pad
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        else:
            token_ids += [self.vocab['<pad>']] * (self.max_seq_len - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        input_seq = self.sentences[idx][:-1]
        target_seq = self.sentences[idx][1:]
        return input_seq, target_seq

# Split into training and validation sets
train_sentences, val_sentences = train_test_split(sentences, test_size=0.1, random_state=42)

train_dataset = CustomDataset(train_sentences, vocab, MAX_SEQ_LEN)
val_dataset = CustomDataset(val_sentences, vocab, MAX_SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# Instantiate model
model = TransformerLanguageModel(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    nhead=NHEAD,
    num_layers=NUM_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=0.1
).to(DEVICE)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# Training function
def train(model, train_loader, val_loader, epochs):
    train_loss_values = []
    val_loss_values = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            src_key_padding_mask = (inputs == vocab['<pad>'])
            optimizer.zero_grad()
            output = model(inputs, src_key_padding_mask)
            loss = criterion(output.view(-1, len(vocab)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_count}, Loss: {loss.item():.4f}")
        average_loss = total_loss / batch_count
        train_loss_values.append(average_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {average_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_batch_count = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                src_key_padding_mask = (inputs == vocab['<pad>'])
                output = model(inputs, src_key_padding_mask)
                loss = criterion(output.view(-1, len(vocab)), targets.view(-1))
                val_loss += loss.item()
                val_batch_count += 1
        val_loss /= val_batch_count
        val_loss_values.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")

        # Update learning rate
        scheduler.step(val_loss)

    # Plot loss curves
    plt.plot(range(1, epochs + 1), train_loss_values, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_loss_values, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()

# Start training
train(model, train_loader, val_loader, EPOCHS)

# Updated text generation function
def generate_text(model, vocab, idx_to_word, max_length=15, temperature=0.8, top_k=5):
    model.eval()
    # Choose a starting word (subject)
    start_words = [word for word in vocab if word.istitle()]
    start_word = random.choice(start_words)
    input_ids = [vocab.get(start_word, vocab['<unk>'])]
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    generated = [start_word]
    with torch.no_grad():
        for _ in range(max_length - 1):
            src_key_padding_mask = (input_tensor == vocab['<pad>'])
            output = model(input_tensor, src_key_padding_mask)
            next_word_logits = output[0, -1, :] / temperature
            next_word_probs = torch.softmax(next_word_logits, dim=0)
            # Avoid zero probability words
            if top_k > 0:
                top_k = min(top_k, next_word_probs.size(-1))
                next_word_probs, top_k_indices = torch.topk(next_word_probs, top_k)
                next_word_probs = next_word_probs / next_word_probs.sum()
                next_word_idx = top_k_indices[torch.multinomial(next_word_probs, 1).item()].item()
            else:
                next_word_idx = torch.multinomial(next_word_probs, 1).item()
            next_word = idx_to_word.get(next_word_idx, '<unk>')
            if next_word == '.':
                generated.append(next_word)
                break
            generated.append(next_word)
            input_ids.append(next_word_idx)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    # Reconstruct the sentence, handling punctuation correctly
    sentence = ''
    for idx, word in enumerate(generated):
        if word in ['.', ',', '?', '!', ';', ':']:
            sentence += word
        elif idx > 0:
            sentence += ' ' + word
        else:
            sentence += word
    return sentence

# Generate sentences
print("\nGenerated Sentences:\n")
for _ in range(10):
    sentence = generate_text(model, vocab, idx_to_word, max_length=15, temperature=0.8, top_k=5)
    print(sentence)
