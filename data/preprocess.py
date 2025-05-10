# import pandas as pd
# import spacy

# def preprocess_text(text, nlp):
#     doc = nlp(text.lower())
#     return " ".join([t.lemma_ for t in doc if not t.is_stop and t.is_alpha])

# def main():
#     nlp = spacy.load("en_core_web_sm")
#     df = pd.read_csv("data/raw/data.csv")
#     df['text'] = df['text'].apply(lambda x: preprocess_text(str(x), nlp))
#     df.to_csv("data/processed/data.csv", index=False)

# if __name__ == "__main__":
#     main()

import pandas as pd
import spacy
import os

def preprocess_text(text, nlp):
    doc = nlp(text.lower())
    return " ".join([t.lemma_ for t in doc if not t.is_stop and t.is_alpha])

def main():
    nlp = spacy.load("en_core_web_sm")

    # Lecture du fichier source
    df = pd.read_csv("data/raw/data.csv")

    # Prétraitement
    df['text'] = df['text'].apply(lambda x: preprocess_text(str(x), nlp))

    # Assurer l'existence du dossier de sortie
    os.makedirs("data/processed", exist_ok=True)

    # Sauvegarde
    df.to_csv("data/processed/data.csv", index=False)

if __name__ == "__main__":
    main()
