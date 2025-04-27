import pandas as pd
import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm.auto import tqdm
import sys
sys.path.append(os.getcwd())
from models.bert import BertClassifier, FakeNewsDataset, FakeNewsPreprocessor

def load_model(model_path):
    """Load a trained BERT model from the specified path"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertClassifier(pretrained_model_name='bert-base-uncased')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def eval_model(model, data_loader, device):
    """Evaluate the model and return all metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate all metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    
    return accuracy, f1, precision, recall

def predict_text(model, tokenizer, text, device, max_len=256):
    """Make prediction for a single text"""
    model.eval()
    with torch.no_grad():
        # Create a temporary DataFrame for preprocessing
        temp_df = pd.DataFrame({'text': [text], 'label': [0]})
        
        # For single element, we'll create the encodings directly
        encodings = tokenizer(
            [text],
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Create dataset directly without splitting
        dataset = FakeNewsDataset(encodings, [0])
        
        # Get the first (and only) item from the dataset
        item = dataset[0]
        input_ids = item['input_ids'].unsqueeze(0).to(device)
        attention_mask = item['attention_mask'].unsqueeze(0).to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        prediction = outputs.argmax(dim=1).item()
        
    return prediction

def main():
    parser = argparse.ArgumentParser(description='Fake News Detection')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model on test data')
    args = parser.parse_args()

    # Load the model and tokenizer
    model_path = 'models/models_trained/bert_fake_news_classifier.pt'
    try:
        model, device = load_model(model_path)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print("Model and tokenizer loaded successfully!")
    except FileNotFoundError:
        print("Error: Model file not found. Please train the model first.")
        return

    if args.evaluate:
        # Load and prepare the data
        try:
            print("Loading data...")
            df_fake = pd.read_csv('data/Fake.csv')
            df_fake['label'] = 0
            df_true = pd.read_csv('data/True.csv')
            df_true['label'] = 1
            df = pd.concat([df_fake, df_true]).reset_index(drop=True)

            # Initialize preprocessor and create dataset
            print("Preprocessing data...")
            preprocessor = FakeNewsPreprocessor(df, tokenizer, max_len=256, val_size=0.1)
            _, test_set = preprocessor.preprocess()
            test_loader = DataLoader(test_set, batch_size=16)

            # Evaluate the model
            print("Evaluating model...")
            accuracy, f1, precision, recall = eval_model(model, test_loader, device)
            print(f"\nModel Evaluation Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
        except FileNotFoundError:
            print("Error: Data files not found. Please ensure True.csv and Fake.csv are in the data directory.")
            return
    else:
        print("\nFake News Detection System")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            text = input("\nEnter a news text to analyze: ")
            if text.lower() == 'quit':
                break
                
            # Make prediction
            prediction = predict_text(model, tokenizer, text, device)
            result = "FAKE" if prediction == 0 else "REAL"
            print(f"\nPrediction: This news is {result}")

if __name__ == "__main__":
    main()


