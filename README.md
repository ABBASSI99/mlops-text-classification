# MLOps Text Classification - Cahier des Charges

## Stack technique
- **Python**
- **DVC**, **MLflow**
- **FastAPI**, **Docker**, **Kubernetes**
- **Prometheus**, **Grafana**
- **OAuth2/JWT**, **TLS**
- **PyTorch/TensorFlow/Scikit-learn**, **Transformers**
- **Design Patterns** : Singleton, Factory, Strategy
- **Générateurs, Itérateurs, Décorateurs**

## Usage
Voir chaque fichier pour l’implémentation détaillée (exemples dans src/).

## Structure du repo
- `src/`
  - gestion données (générateurs/itérateurs)
  - modèles (factory/strategy/singleton)
  - entraînement & tracking
  - API FastAPI (déploiement, sécurité)
  - monitoring
- `dvc.yaml`, `params.yaml`
- `Dockerfile`, `docker-compose.yaml`
- `.github/workflows/ci.yml`