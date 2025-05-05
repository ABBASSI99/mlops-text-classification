from prometheus_client import Counter, Gauge

REQUESTS = Counter("api_requests_total", "Nombre total de requêtes API")
INFERENCES = Counter("model_inferences_total", "Nombre d'inférences")
DRIFT_ALERT = Gauge("model_drift_alert", "Drift détecté (1=oui, 0=non)")

def inc_request():
    REQUESTS.inc()

def inc_inference():
    INFERENCES.inc()

def set_drift(alert):
    DRIFT_ALERT.set(1 if alert else 0)