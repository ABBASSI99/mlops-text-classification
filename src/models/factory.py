from src.models.logreg import LogisticRegressionModel
from src.models.lstm import LSTMModel
from src.models.bert import BERTModel

class ModelFactory:
    @staticmethod
    def create_model(model_type, **kwargs):
        if model_type == "logreg":
            return LogisticRegressionModel(**kwargs)
        elif model_type == "lstm":
            return LSTMModel(**kwargs)
        elif model_type == "bert":
            return BERTModel(**kwargs)
        else:
            raise ValueError("Unknown model type")