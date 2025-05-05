import csv

# Générateur pour lecture paresseuse des données
def data_generator(path, text_col='text', label_col='label'):
    with open(path, 'r', encoding='utf8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row[text_col], row[label_col]

# Itérateur pour batcher les données
class DataBatchIterator:
    def __init__(self, generator, batch_size=32):
        self.generator = generator
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for item in self.generator:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch