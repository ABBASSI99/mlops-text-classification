from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from src.models.singleton import ModelSingleton
from src.models.factory import ModelFactory
from src.security.rbac import get_current_user, has_permission
from src.monitoring.metrics import inc_request, inc_inference
from prometheus_client import start_http_server

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class PredictRequest(BaseModel):
    text: str

@app.on_event("startup")
async def load_model():
    # Charger le modèle au démarrage
    model = ModelFactory.create_model("logreg")  # Par défaut, régression logistique
    ModelSingleton(model)
    start_http_server(8001)  # Expose les métriques sur le port 8001

@app.post("/predict")
async def predict(
    request: PredictRequest,
    token: str = Depends(oauth2_scheme)
):
    inc_request()
    user = get_current_user(token)
    if not has_permission(user, "predict"):
        raise HTTPException(status_code=403, detail="Permission denied")
    model = ModelSingleton._instance.model
    pred = model.predict([request.text])
    inc_inference()
    return {"prediction": pred}