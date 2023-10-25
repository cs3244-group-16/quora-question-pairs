from os.path import join

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.functional as F
from transformers import DistilBertModel, DistilBertTokenizer

np.random.seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_DIR = join('..', 'data', 'raw')
train_df = pd.read_csv(join(DATA_DIR, 'train.csv'))
train_df.head()
train_df = train_df.fillna("") 

SEP = '[SEP]'
train_df['question1'] = train_df['question1'].str.lower()
train_df['question2'] = train_df['question2'].str.lower()
train_df['concat_qns'] = train_df['question1'] + ' ' + SEP + ' ' + train_df['question2']

train_df.loc[0, 'concat_qns']


MAX_LEN = 512
BATCH_SIZE = 16

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', return_dict=False)

class QuoraDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        qns = str(self.X.iloc[index]['concat_qns'])
        encoded_qns = self.tokenizer.encode_plus(
            qns,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
        )
        ids = encoded_qns['input_ids']
        mask = encoded_qns['attention_mask']
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'target': torch.tensor(self.y[index])
        }

    def __len__(self):
        return len(self.X)

X = train_df[['concat_qns']]
y = train_df['is_duplicate']
y = torch.tensor(y, dtype=torch.float32)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)


train_dataset = QuoraDataset(X_train, y_train, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = QuoraDataset(X_val, y_val, tokenizer, MAX_LEN)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = QuoraDataset(X_val, y_val, tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


class DistilBertClass(torch.nn.Module):
    def __init__(self, dropout=0.1):
        super(DistilBertClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=False)
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(768, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.sigmoid(output)
        return output

def train(model, criterion, optimizer, scheduler=None, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            ids = batch['ids'].to(device, dtype = torch.long)
            mask = batch['mask'].to(device, dtype = torch.long)
            labels = batch['target'].to(device, dtype = torch.float)

            optimizer.zero_grad()
            outputs = model(ids, attention_mask=mask)
            loss = criterion(outputs.view(-1), labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        if scheduler: scheduler.step()
        print(f"Epoch {epoch+1}: loss = {total_loss:.2f}")
    return model


def evaluate(model):
    model.eval()
    with torch.no_grad():
        # train set
        correct = 0
        total = 0
        for batch in tqdm(train_loader):
            ids = batch['ids'].to(device, dtype = torch.long)
            mask = batch['mask'].to(device, dtype = torch.long)
            labels = batch['target'].to(device, dtype = torch.float)

            outputs = model(ids, attention_mask=mask)
            predicted = (outputs > 0.5).float().view(-1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Train Accuracy: {accuracy:.2f}%")

        # validation set
        correct = 0
        total = 0
        for batch in tqdm(val_loader):
            ids = batch['ids'].to(device, dtype = torch.long)
            mask = batch['mask'].to(device, dtype = torch.long)
            labels = batch['target'].to(device, dtype = torch.float)

            outputs = model(ids, attention_mask=mask)
            predicted = (outputs > 0.5).float().view(-1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")

model = DistilBertClass()
model.to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-5)
model = train(model, criterion, optimizer, num_epochs=2)

evaluate(model)
evaluate(model)

output_model_file = '../models/pytorch_distilbert.bin'
output_vocab_file = '../models/vocab_distilbert.bin'
torch.save(model, output_model_file)
tokenizer.save_vocabulary(output_vocab_file)
print('Model saved')
