# import mlflow
# from data.loader import data_generator
# from src.models.factory import ModelFactory
# from src.decorators.logging import log_time
# from sklearn.model_selection import train_test_split

# @log_time
# def train(model_type, data_path):
#     mlflow.start_run()
#     gen = data_generator(data_path)
#     X, y = [], []
#     for text, label in gen:
#         X.append(text)
#         y.append(label)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = ModelFactory.create_model(model_type)
#     model.fit(X_train, y_train)
#     accuracy = model.evaluate(X_test, y_test)
#     mlflow.log_param("model_type", model_type)
#     mlflow.log_metric("accuracy", accuracy)
#     mlflow.end_run()
#     return model  # Retourne le modèle pour le singleton

# if __name__ == "__main__":
#     train("logreg", "data/processed/data.csv")


import mlflow
from data.loader import data_generator
from src.models.factory import ModelFactory
from src.decorators.logging import log_time
from sklearn.model_selection import train_test_split

@log_time
def train(model_type, data_path):
    mlflow.start_run()
    gen = data_generator(data_path)
    X, y = [], []
    for text, label in gen:
        X.append(text)
        y.append(label)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = ModelFactory.create_model(model_type)
    model.fit(X_train, y_train)
    accuracy = model.evaluate(X_test, y_test)

    mlflow.log_param("model_type", model_type)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.end_run()

    return model

if __name__ == "__main__":
    import sys
    model_type = sys.argv[1]
    data_path = sys.argv[2]
    train(model_type, data_path)
