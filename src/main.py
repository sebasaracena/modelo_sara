from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from train import RUTA_MODELO

app = FastAPI()

Dummy = Annotated[int, Field(ge=0, le=1)]


class PrediccionRequest(BaseModel):
    ano_carrera_actual: int
    dep_cs_hm: Dummy
    dep_sd: Dummy
    nota_parcial_1: float
    n_estudiantes_encurso: int
    porcen_aproba_curso: float
    semestre_academico: int
    puntaje_ranking: float
    tipo_ingreso_bea: Dummy
    tipo_ingreso_pace: Dummy
    tipo_ingreso_especial: Dummy
    tipo_ingreso_tranfexterna: Dummy
    promedio_notas_estudiante: float
    progresion_academica_sct: float
    sct_curso: float


class PrediccionResponse(BaseModel):
    riesgo_reprobar: float
    prediccion: int


_estado = {"model": None, "columnas_x": None, "media_ano_ingreso": None}


def cargar_modelo():
    import joblib

    payload = joblib.load(RUTA_MODELO)
    _estado["model"] = payload["model"]
    _estado["columnas_x"] = payload["columnas_x"]
    _estado["media_ano_ingreso"] = payload["media_ano_ingreso"]


try:
    cargar_modelo()
except FileNotFoundError:
    pass


@app.get("/health")
def health():
    if _estado["model"] is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return {"status": "ok"}


@app.post("/predict", response_model=PrediccionResponse)
def predict(request: PrediccionRequest):
    if _estado["model"] is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    valores = {
        "const": 1.0,
        "ano_carrera_actual": request.ano_carrera_actual - _estado["media_ano_ingreso"],
    }
    for campo in type(request).model_fields:
        if campo != "ano_carrera_actual":
            valores[campo] = getattr(request, campo)

    exog = np.array([[valores[c] for c in _estado["columnas_x"]]])
    prob_aprobar = _estado["model"].predict(exog=exog)[0]
    riesgo_reprobar = 1.0 - prob_aprobar
    # 0.35 sobre riesgo_reprobar == 0.65 sobre prob_aprobar en evaluar_modelo() (modelo5logit.py): mismo corte, ambos lados de 1 - prob_aprobar
    prediccion = 1 if riesgo_reprobar >= 0.35 else 0
    return PrediccionResponse(riesgo_reprobar=float(riesgo_reprobar), prediccion=prediccion)
