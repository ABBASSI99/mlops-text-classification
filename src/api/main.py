# from fastapi import FastAPI, Depends, HTTPException
# from fastapi.security import OAuth2PasswordBearer
# from pydantic import BaseModel
# from src.models.singleton import ModelSingleton
# from src.models.factory import ModelFactory
# from src.security.rbac import get_current_user, has_permission
# from src.monitoring.metrics import inc_request, inc_inference
# from prometheus_client import start_http_server

# app = FastAPI()
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# class PredictRequest(BaseModel):
#     text: str

# @app.on_event("startup")
# async def load_model():
#     # Charger le modèle au démarrage
#     model = ModelFactory.create_model("logreg")  # Par défaut, régression logistique
#     ModelSingleton(model)
#     start_http_server(8001)  # Expose les métriques sur le port 8001

# @app.post("/predict")
# async def predict(
#     request: PredictRequest,
#     token: str = Depends(oauth2_scheme)
# ):
#     inc_request()
#     user = get_current_user(token)
#     if not has_permission(user, "predict"):
#         raise HTTPException(status_code=403, detail="Permission denied")
#     model = ModelSingleton._instance.model
#     pred = model.predict([request.text])
#     inc_inference()
#     return {"prediction": pred}

# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# async def read_root():
#     return {"message": "Hello, World!"}


from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from src.models.singleton import ModelSingleton
from src.models.factory import ModelFactory
from src.security.rbac import get_current_user, has_permission
from src.monitoring.metrics import inc_request, inc_inference
from prometheus_client import start_http_server
from typing import Optional, List, Dict

app = FastAPI(
    title="API de Classification de Texte",
    description="API pour classifier du texte en utilisant un modèle de machine learning.",
    version="1.0.0"
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

class PredictRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confiance: float
    autres_categories: Dict[str, float]

class Token(BaseModel):
    access_token: str
    token_type: str

@app.on_event("startup")
async def load_model():
    # Charger le modèle au démarrage
    model = ModelFactory.create_model("logreg")  # Par défaut, régression logistique
    ModelSingleton(model)
    start_http_server(8001)  # Expose les métriques sur le port 8001

@app.get("/", summary="Accueil de l'API")
async def read_root():
    return {"message": "Bienvenue sur l'API de classification de texte"}

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # For testing purposes, accept any username/password
    return {"access_token": "test_token", "token_type": "bearer"}

@app.post("/predict", 
         summary="Prédire la classe d'un texte", 
         tags=["Prediction"],
         response_model=PredictionResponse)
async def predict(
    request: PredictRequest,
    token: Optional[str] = Security(oauth2_scheme)
):
    inc_request()
    try:
        model = ModelSingleton._instance.model
        predictions = model.predict([request.text])
        inc_inference()
        return predictions[0]  # Return the first prediction since we only process one text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
