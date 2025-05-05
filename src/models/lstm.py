import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_encoder):
        self.texts = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in texts]
        self.labels = label_encoder.transform(labels)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.texts[idx], self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    padded = pad_sequence(texts, batch_first=True)
    return padded, torch.tensor(labels)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

class LSTMModel:
    def __init__(self, tokenizer, num_classes=2):
        self.tokenizer = tokenizer
        self.label_encoder = LabelEncoder()
        self.model = LSTMClassifier(vocab_size=tokenizer.vocab_size, num_classes=num_classes)
    def fit(self, X, y):
        self.label_encoder.fit(y)
        dataset = TextDataset(X, y, self.tokenizer, self.label_encoder)
        loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(3):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = loss_fn(out, batch_y)
                loss.backward()
                optimizer.step()
    def predict(self, X):
        self.model.eval()
        dataset = TextDataset(X, [0]*len(X), self.tokenizer, self.label_encoder)
        loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
        preds = []
        with torch.no_grad():
            for batch_x, _ in loader:
                out = self.model(batch_x)
                preds += out.argmax(1).cpu().numpy().tolist()
        return self.label_encoder.inverse_transform(preds).tolist()
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.mean(np.array(y_pred) == np.array(y))