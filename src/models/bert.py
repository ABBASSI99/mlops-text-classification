from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np

class BERTModel:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    def fit(self, X, y):
        # Pour un vrai projet, adapter le Dataset à HuggingFace Dataset pour Trainer
        from datasets import Dataset
        ds = Dataset.from_dict({'text': X, 'label': y})
        def tokenize(batch): return self.tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)
        ds = ds.map(tokenize, batched=True)
        args = TrainingArguments('bert-out', per_device_train_batch_size=8, num_train_epochs=1, logging_steps=10)
        trainer = Trainer(model=self.model, args=args, train_dataset=ds)
        trainer.train()
    def predict(self, X):
        tokens = self.tokenizer(X, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        outputs = self.model(**tokens)
        return outputs.logits.argmax(-1).tolist()
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.mean(np.array(y_pred) == np.array(y))