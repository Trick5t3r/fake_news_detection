import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch import nn
from tqdm.auto import tqdm

import os
import sys
sys.path.append(os.getcwd())
from models.bert import BertClassifier, FakeNewsDataset, FakeNewsPreprocessor


# 4. Fonctions d'entraînement et d'évaluation

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, num_epochs, epoch):
    model.train()
    total_loss, correct = 0, 0
    for batch in tqdm(data_loader, desc=f"Training {epoch+1}/{num_epochs}", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, labels)
        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total_loss += loss.detach().item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()

    acc = correct / len(data_loader.dataset)
    return acc, total_loss / len(data_loader)


def eval_model(model, data_loader, loss_fn, device, num_epochs, epoch):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {epoch+1}/{num_epochs}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total_loss += loss.detach().item()

    acc = correct / len(data_loader.dataset)
    return acc, total_loss / len(data_loader)

# Pipeline principale
def main():
    # Hyperparamètres
    PRETRAINED_MODEL_NAME = 'bert-base-uncased'
    MAX_LEN = 256; BATCH_SIZE = 16; EPOCHS = 3; LR = 2e-5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Chargement des deux CSV et création des labels
    print("Loading data...")
    df_fake = pd.read_csv('data/Fake.csv')
    df_fake['label'] = 0
    df_true = pd.read_csv('data/True.csv')
    df_true['label'] = 1
    df = pd.concat([df_fake, df_true]).reset_index(drop=True)

    # Initialisation du tokenizer et du préprocesseur
    print("Initializing tokenizer and preprocessor...")
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    preprocessor = FakeNewsPreprocessor(df, tokenizer, max_len=MAX_LEN, val_size=0.1)
    train_set, val_set = preprocessor.preprocess()

    # DataLoaders
    print("Creating DataLoaders...")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)

    # Modèle, optimizer, scheduler, loss
    print("Initializing model...")
    model = BertClassifier(pretrained_model_name=PRETRAINED_MODEL_NAME)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    # Entraînement
    print("Training model...")
    for epoch in range(EPOCHS):
        train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler, EPOCHS, epoch)
        val_acc, val_loss     = eval_model(model, val_loader, loss_fn, device, EPOCHS, epoch)
        print(f'Epoch {epoch+1}/{EPOCHS} — '
              f'Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | '
              f'Val loss: {val_loss:.4f}, acc: {val_acc:.4f}')

    # Sauvegarde du modèle
    print("Saving model...")
    torch.save(model.state_dict(), 'models/models_trained/bert_fake_news_classifier.pt')

if __name__ == '__main__':
    main()