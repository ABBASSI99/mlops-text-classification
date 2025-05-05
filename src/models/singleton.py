# Singleton pour la gestion centralisée du modèle (ex : pour inference sur API)
class ModelSingleton:
    _instance = None

    def __new__(cls, model):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = model
        return cls._instance