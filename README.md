# coeficientes_SARA

Modelo de riesgo academico estudiantil (Logit) entrenado sobre `BD_SARA_vf.xlsx`.
Los coeficientes resultantes se copian a mano a `ecuacionlogit.md`, que es el
codigo real que corre en Dagster (`_calcular_ado`) para calcular
`p_riesgo_SARA` en produccion.

## Como correr

- **Reentrenar modelo** (genera `assets/modelo_logit.pkl`): `python src/train.py`
- **Sacar coeficientes / accuracy / precision / recall** (genera
  `assets/coeficientes_modelo5logit.csv` y matriz de confusion):
  `python src/modelo5logit.py`
- **Levantar API FastAPI local** (desde `src/`): `uvicorn main:app --reload`
- **Correr tests**: `pytest tests/`

## Checklist: pasos manuales despues de reentrenar

El entrenamiento (`python src/train.py` o `python src/modelo5logit.py`) es
automatico — lee el Excel tal cual este y entrena sin necesidad de tocar
codigo, incluso si aparecen columnas `Carrera_curso_*` / `Tipo_ingreso_*`
nuevas (categorias nuevas).

Pero **no hay sync automatico hacia produccion**. Despues de reentrenar, hacer
a mano:

- [ ] Copiar los coeficientes nuevos (`assets/coeficientes_modelo5logit.csv`)
      a la formula `_calcular_ado` en `ecuacionlogit.md`.
- [ ] Si hay categoria nueva (carrera o tipo de ingreso), agregar su linea
      correspondiente en `_calcular_ado` — no aparece sola.
- [ ] Si hay categoria nueva, agregarla tambien a `CAMPO_A_COLUMNA` en
      `src/main.py` (API local de pruebas), sino da `KeyError` al usarla.
- [ ] Confirmar que el pipeline Dagster (fuera de este repo) manda el campo
      nuevo en el `fila` dict que llega a `_calcular_ado`.
- [ ] Aplicar siempre la sigmoide sobre `z_score`, nunca usar `z` crudo como
      riesgo (`riesgo = 1 - sigmoid(z)`, ver nota en `ecuacionlogit.md`).
- [ ] Centrar `Año_Ingreso` en produccion con la MISMA media que uso el
      entrenamiento (valor actual: 2019.47) — no recalcular con datos nuevos.

## Cosas a tener en cuenta

- **Categoria de referencia (base):** al entrenar, se saca una columna por
  cada grupo de dummies (`Carrera_curso_*`, `Tipo_ingreso_*`) para evitar la
  trampa de variable dummy (colinealidad perfecta). La que queda de base es
  simplemente la primera columna del grupo en el orden del Excel — no importa
  cual sea, el riesgo final da identico, solo cambia que tan legibles quedan
  los coeficientes individuales.
- **Categoria nueva con pocos datos:** un coeficiente estimado con pocas filas
  (ej. `Carrera_Curso_icinf` con 58 filas vs 300-11000 del resto) sale mas
  inestable/ruidoso (p-value alto). Esperable las primeras corridas, se
  estabiliza con mas datos con el tiempo — no es señal de bug.
- **Escala `promedio_sct`:** el modelo entrena en escala 0-100
  (`Progresión_académica_SCT`), pero `indicador_sara.csv` (Dagster) trae
  fraccion 0-1. La ecuacion ya escala (`v("promedio_sct") * 100`); si se
  cambia el origen del dato, revisar este punto.
- **Columna con espacio final:** `BD_SARA_vf.xlsx` trae
  `"Promedio_notas_estudiante "` (espacio al final). Si se toca
  `CAMPO_A_COLUMNA` o se agregan columnas nuevas, usar el nombre EXACTO tal
  como esta en el Excel.
- **No sacar `Promedio_notas_estudiante` del entrenamiento:** aunque esta
  correlacionada con `Progresion_academica_SCT` (colinealidad, corr 0.935),
  sacarla ya se probo y se revirtio — empeoraba mucho el riesgo calculado
  para alumnos recien ingresados (avance=0).


## Estructura

- `src/modelo5logit.py` — modelo Logit + fixes (dummy trap, centrado de año,
  columnas excluidas). `src/train.py` — reentrena y persiste
  `assets/modelo_logit.pkl` (uso: QA local via `src/main.py`, API FastAPI,
  no reemplaza Dagster).
- `ecuacionlogit.md` — ecuacion real en produccion (Dagster).
- `ECUACION_MD.md`, `CHANGES_LOGIT.md`, `Resumen_analisis.md` — bitacoras de
  por que la ecuacion cambio historicamente.
- `MEJORAS_MODELO.md` — historial cronologico completo de mejoras al
  modelo: trampa dummy, sigmoide, escalas/centrado, colinealidad, migracion
  Excel a CSV (90.97% a 93.29% accuracy), y analisis accuracy vs recall por
  clase (matriz de confusion, threshold pendiente de aplicar).
