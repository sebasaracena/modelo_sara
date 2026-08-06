import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

# Fila real (indice 0) de assets/modelo_sara.csv, en formato del payload de la API.
PAYLOAD_VALIDO = {
    "ano_carrera_actual": 2017,
    "dep_cs_hm": 1,
    "dep_sd": 0,
    "nota_parcial_1": 3.5,
    "n_estudiantes_encurso": 14,
    "porcen_aproba_curso": 94.0,
    "semestre_academico": 2,
    "puntaje_ranking": 533.29,
    "tipo_ingreso_bea": 0,
    "tipo_ingreso_pace": 0,
    "tipo_ingreso_especial": 1,
    "tipo_ingreso_tranfexterna": 0,
    "promedio_notas_estudiante": 4.7,
    "progresion_academica_sct": 91.0,
    "sct_curso": 3.0,
}


def test_predict_ok():
    response = client.post("/predict", json=PAYLOAD_VALIDO)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["riesgo_reprobar"] <= 1.0
    assert data["prediccion"] in (0, 1)
    assert data["prediccion"] == (1 if data["riesgo_reprobar"] >= 0.5 else 0)


def test_predict_campo_faltante():
    payload = dict(PAYLOAD_VALIDO)
    del payload["nota_parcial_1"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_campo_null():
    payload = dict(PAYLOAD_VALIDO)
    payload["nota_parcial_1"] = None
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_tipo_incorrecto():
    payload = dict(PAYLOAD_VALIDO)
    payload["nota_parcial_1"] = "no-es-numero"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_json_malformado():
    response = client.post(
        "/predict",
        content=b"{esto no es json valido",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_predict_dummy_fuera_de_rango():
    payload = dict(PAYLOAD_VALIDO)
    payload["dep_sd"] = 2
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_metodo_incorrecto():
    response = client.get("/predict")
    assert response.status_code == 405


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200


def test_riesgo_alto_prediccion_1(monkeypatch):
    import main

    monkeypatch.setattr(main._estado["model"], "predict", lambda exog: [0.1])
    response = client.post("/predict", json=PAYLOAD_VALIDO)
    assert response.status_code == 200
    data = response.json()
    assert data["riesgo_reprobar"] == pytest.approx(0.9)
    assert data["prediccion"] == 1


def test_riesgo_bajo_prediccion_0(monkeypatch):
    import main

    monkeypatch.setattr(main._estado["model"], "predict", lambda exog: [0.9])
    response = client.post("/predict", json=PAYLOAD_VALIDO)
    assert response.status_code == 200
    data = response.json()
    assert data["riesgo_reprobar"] == pytest.approx(0.1)
    assert data["prediccion"] == 0
