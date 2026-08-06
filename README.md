# coeficientes_SARA

Modelo de riesgo academico estudiantil (Logit) entrenado sobre
`assets/modelo_sara.csv`. Los coeficientes resultantes se copian a mano a
[`ecuacionlogit.md`](ecuacionlogit.md), que es el codigo real que corre en Dagster
(`_calcular_ado`) para calcular `p_riesgo_SARA` en produccion.

## Como correr

- **Reentrenar modelo** (genera `assets/modelo_logit.pkl`): `python src/train.py`
- **Sacar coeficientes / accuracy / precision / recall** (genera
  `assets/coeficientes_modelo5logit.csv` y matriz de confusion):
  `python src/modelo5logit.py`
- **Levantar API FastAPI local** (desde `src/`): `uvicorn main:app --reload`
- **Correr tests**: `pytest tests/`

## Checklist: pasos manuales despues de reentrenar

El entrenamiento (`python src/train.py` o `python src/modelo5logit.py`) es
automatico — lee `assets/modelo_sara.csv` tal cual este y entrena sin
necesidad de tocar codigo, incluso si aparecen columnas `dep_*` /
`tipo_ingreso_*` nuevas (categorias nuevas).

Pero **no hay sync automatico hacia produccion**. Despues de reentrenar, hacer
a mano:

- [ ] Copiar los coeficientes nuevos (`assets/coeficientes_modelo5logit.csv`)
      a la formula `_calcular_ado` en [`ecuacionlogit.md`](ecuacionlogit.md).
- [ ] Si hay categoria nueva (departamento o tipo de ingreso), agregar su
      linea correspondiente en `_calcular_ado` — no aparece sola.
- [ ] Si hay categoria/campo nuevo, agregarlo tambien al modelo
      `PrediccionRequest` en `src/main.py` (API local de pruebas), sino no
      llega al `exog` que se le pasa al modelo.
- [ ] Confirmar que el pipeline Dagster (fuera de este repo) manda el campo
      nuevo en el `fila` dict que llega a `_calcular_ado`.
- [ ] Aplicar siempre la sigmoide sobre `z_score`, nunca usar `z` crudo como
      riesgo (`riesgo = 1 - sigmoid(z)`, ver nota en [`ecuacionlogit.md`](ecuacionlogit.md)).
- [ ] Centrar `ano_carrera_actual` en [`ecuacionlogit.md`](ecuacionlogit.md) con la MISMA media
      que uso el entrenamiento (valor actual: 2021.02, ver constante
      hardcodeada en `_calcular_ado`) — no recalcular con datos nuevos. En
      `src/main.py` esto es automatico: la media se guarda en
      `modelo_logit.pkl` al entrenar y se lee sola.

## Cosas a tener en cuenta

- **Categoria de referencia (base):** al entrenar, se saca una columna por
  cada grupo de dummies (`dep_*`, `tipo_ingreso_*`) para evitar la trampa de
  variable dummy (colinealidad perfecta). La que queda de base es simplemente
  la primera columna del grupo en el orden del CSV — no importa cual sea, el
  riesgo final da identico, solo cambia que tan legibles quedan los
  coeficientes individuales.
- **Categoria nueva con pocos datos:** un coeficiente estimado con pocas filas
  sale mas inestable/ruidoso (p-value alto). Esperable las primeras corridas
  de una categoria nueva, se estabiliza con mas datos con el tiempo — no es
  señal de bug.
- **Escala `progresion_academica_sct`:** el modelo entrena en escala 0-100
  sobre `modelo_sara.csv`. Si el dato de entrada en produccion viene en otra
  escala (ej. fraccion 0-1), hay que escalar antes de usar la ecuacion.
- **No sacar `promedio_notas_estudiante` del entrenamiento:** aunque esta
  correlacionada con `progresion_academica_sct` (colinealidad, corr 0.741),
  sacarla ya se probo y se revirtio — empeoraba mucho el riesgo calculado
  para alumnos recien ingresados (avance=0).


## Estructura

- `src/modelo5logit.py` — modelo Logit + fixes (dummy trap, centrado de año,
  columnas excluidas). `src/train.py` — reentrena y persiste
  `assets/modelo_logit.pkl` (uso: QA local via `src/main.py`, API FastAPI,
  no reemplaza Dagster).
- [`ecuacionlogit.md`](ecuacionlogit.md) — ecuacion real en produccion (Dagster).
- [`MEJORAS_MODELO.md`](MEJORAS_MODELO.md) — historial cronologico completo
  de mejoras al modelo: trampa dummy, sigmoide, escalas/centrado,
  colinealidad, y analisis accuracy vs recall por clase (matriz de
  confusion, threshold pendiente de aplicar).
