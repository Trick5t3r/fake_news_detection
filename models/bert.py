import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertModel
from sklearn.model_selection import train_test_split

# 1. Classe de preprocessing
class FakeNewsPreprocessor:
    def __init__(self, df, tokenizer, max_len=512, val_size=0.1,
                 feature_name='text', label_name='label'):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.val_size = val_size
        self.feature_name = feature_name
        self.label_name = label_name

    def preprocess_text(self, text):
        # Personnalisation du texte (ex: lowercasing)
        return str(text).lower()

    def preprocess(self):
        # Appliquer preprocessing textuel
        texts = self.df[self.feature_name].apply(self.preprocess_text).tolist()
        labels = self.df[self.label_name].tolist()

        # Tokenisation en batch
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Split train / validation
        train_idx, val_idx = train_test_split(
            range(len(labels)),
            test_size=self.val_size,
            random_state=42,
            stratify=labels
        )

        train_enc = {k: v[train_idx] for k, v in encodings.items()}
        val_enc   = {k: v[val_idx]   for k, v in encodings.items()}
        train_labels = [labels[i] for i in train_idx]
        val_labels   = [labels[i] for i in val_idx]

        train_set = FakeNewsDataset(train_enc, train_labels)
        val_set   = FakeNewsDataset(val_enc,   val_labels)
        return train_set, val_set

# 2. Dataset pour PyTorch
class FakeNewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# 3. Modèle de classification BERT
class BertClassifier(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', n_classes=2, dropout_prob=0.3):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.drop = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped = self.drop(pooled_output)
        return self.fc(dropped)