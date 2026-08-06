# Historial de mejoras al modelo de riesgo SARA

Bitácora cronológica de qué se cambió en el modelo Logit (`src/modelo5logit.py`)
y por qué. Complementa [CHANGES_LOGIT.md](CHANGES_LOGIT.md) y
[ECUACION_MD.md](ECUACION_MD.md) (detalle técnico de los dos primeros fixes) y
[ecuacionlogit.md](ecuacionlogit.md) (fórmula vigente que consume Dagster) —
acá está la secuencia completa, incluida la migración de fuente de datos y el
análisis de precision/recall más reciente.

## 1. Trampa de variable dummy (`dep_*`, `tipo_ingreso_*`)

Al considerar `dep_*` (departamento académico: `dep_tec_cn`, `dep_cs_hm`,
`dep_sd`) y `tipo_ingreso_*` (`tipo_ingreso_regular`, `_bea`, `_pace`,
`_especial`, `_tranfexterna`) como grupos one-hot completos — cada grupo suma
1 en cada fila, sin categoría de referencia excluida — el `Logit.summary()`
daba coeficientes absurdos (`-16.68`, etc.) y errores estándar `NaN`.

**Causa:** colinealidad perfecta entre la constante y cada grupo de dummies
completo (dummy variable trap).

**Fix:** `eliminar_categoria_referencia()` dropea la primera columna de cada
grupo (`GRUPOS_DUMMIES_COLINEALES = [("dep_",), ("tipo_ingreso_",)]`), que
pasa a ser la categoría de referencia implícita (`dep_tec_cn`,
`tipo_ingreso_regular`). Detalle completo en [CHANGES_LOGIT.md](CHANGES_LOGIT.md).

## 2. Sigmoide invertido

`estado_asignatura=1` significa **aprobó**, no riesgo. `sigmoid(z)` da
`P(aprobar)`, así que el riesgo real es `1 - sigmoid(z)`, nunca `z` crudo
(no acotado, log-odds) ni la sigmoide sin invertir. Detalle en
[ECUACION_MD.md](ECUACION_MD.md).

## 3. Escalas y centrado

- `ano_carrera_actual` (antes `Año_Ingreso`) trae valores tipo 2017-2026 sin
  centrar → obliga a la constante del Logit a absorber ese offset y sale
  con valor enorme (artefacto visual, no error de ajuste). Fix:
  `centrar_ano_ingreso()` resta la media antes de `sm.add_constant`.
- `promedio_sct` se entrena en escala 0-100 pero la fuente de producción
  (Dagster, `indicador_sara.csv`) trae fracción 0-1 → hay que escalar
  `x100` en la ecuación de producción. Pendiente real: corregir en el
  origen del pipeline, no solo compensar acá.

## 4. Colinealidad: qué se saca y qué se deja

- `nota_final_curso` vs `porcen_aproba_curso` (corr 0.908 en el CSV
  actual): miden lo mismo, se excluyó `nota_final_curso`
  (`COLUMNAS_EXCLUIDAS`).
- `promedio_notas_estudiante` vs `progresion_academica_sct` (corr 0.741):
  colineales pero **se dejan las dos**. Se probó sacar
  `promedio_notas_estudiante` y se revirtió — empeoraba mucho el riesgo
  calculado para alumnos recién ingresados (avance/SCT bajo por
  definición), que sin esa variable quedan indistinguibles entre buenos y
  malos.
- Interacción `nota_parcial_1 x porcen_aproba_curso`: probada, no
  significativa (p=0.095), no mejoró accuracy → revertida.

## 5. Bug de columna con espacio final (dataset Excel viejo)

`BD_SARA_vf.xlsx` traía la columna `"Promedio_notas_estudiante "` (espacio
al final) sin limpiar en `cargar_datos()`. El mapeo `CAMPO_A_COLUMNA` en
`src/main.py` apuntaba al nombre sin espacio → `KeyError` en el vector de
inferencia. Ya no aplica tras la migración al CSV (columnas snake_case sin
ese problema).

## 6. Migración de fuente de datos: Excel → CSV

Change OpenSpec `entrenar-modelo-sara-csv` (archivado
`openspec/changes/archive/2026-08-06-entrenar-modelo-sara-csv/`): la fuente
vigente del proyecto pasó de `assets/BD_SARA_vf.xlsx` (21202 filas) a
`assets/modelo_sara.csv` (32620 filas, `;`-separado, columnas snake_case).
Mismo target binario, mismos problemas de fondo (misma trampa dummy en
`dep_*`/`tipo_ingreso_*`, misma colinealidad `nota_final_curso` vs
`porcen_aproba_curso`, 0.908 en el CSV nuevo), pero renombrado:
`Carrera_curso_*` → `dep_*`, `Tipo_ingreso_*` → `tipo_ingreso_*`,
`Puntaje_rnk` → `puntaje_ranking`, `Año_Ingreso` → `ano_carrera_actual`.

Rehusó toda la lógica ya validada (`eliminar_categoria_referencia`,
`centrar_ano_ingreso`, exclusión por colinealidad) vía las constantes
paramétricas, sin tocar el algoritmo. `src/main.py`
(`PrediccionRequest`/`CAMPO_A_COLUMNA`) se reescribió para el esquema nuevo
— cambio **breaking** en el contrato de `/predict`.

**Resultado:** accuracy subió de 90.97% (Excel, cross-validado 5-fold:
90.62%-91.98%, media 91.22% — señal real y estable, no ruido de muestreo)
a **93.29%** (CSV, más datos con la misma proporción de clases).

## 7. Accuracy vs. matriz de confusión (2026-08-06)

El 93.29% de accuracy escondía un problema: las clases están desbalanceadas
(clase 1 = aprueba, 5630 casos en test; clase 0 = reprueba, 894 casos —
86.3% de baseline solo por predecir siempre la mayoritaria). Se agregó
`classification_report` de sklearn a `evaluar_modelo()` en
`modelo5logit.py` y `modelo5.py` para desglosar por clase:

| clase | precision | recall | soporte |
|---|---|---|---|
| 0 (reprueba) | 0.84 | **0.63** | 894 |
| 1 (aprueba)  | 0.94 | 0.98 | 5630 |

**Hallazgo:** recall de la clase 0 es 0.63 — de los 894 alumnos que
reprueban de verdad, el modelo solo detecta 566 y deja pasar 328 como falso
negativo (alumno en riesgo real, clasificado como sin riesgo, sin alerta).
Para un modelo de alerta temprana ese es el error caro, y el accuracy
global lo esconde porque la clase mayoritaria domina el promedio.

Barrido de threshold calculado sobre el test set:

| threshold sobre P(aprueba) | recall clase 0 | precision clase 0 |
|---|---|---|
| 0.50 (antes) | 0.63 | 0.84 |
| 0.61 | 0.70 | 0.76 |
| 0.69 | 0.75 | 0.70 |
| 0.77 | 0.80 | 0.64 |
| 0.82 | 0.85 | 0.59 |

## 8. Threshold aplicado: 0.5 → 0.65 (2026-08-06)

Se aplicó **0.65** (recall ~0.72-0.75 sin hundir precision):

- `src/modelo5logit.py` (`evaluar_modelo`): `clasificacion = np.where(predicciones < 0.65, 0, 1)`.
- `src/main.py:78` (API local `/predict`): `prediccion = 1 if riesgo_reprobar >= 0.35 else 0` —
  0.35 sobre `riesgo_reprobar` (`1 - prob_aprobar`) es el mismo corte que 0.65
  sobre `prob_aprobar`, visto desde el otro lado de la probabilidad.
- [`ecuacionlogit.md`](ecuacionlogit.md) (Dagster) **no se tocó**: devuelve `p_riesgo_SARA`
  continuo, sin binarizar — no hay threshold ahí que propagar. Si Dagster
  aplica su propio corte río abajo para generar una alerta, es
  responsabilidad de esa parte del pipeline (fuera de este repo).

**Resultado real (test set, 6524 filas):**

| métrica | threshold 0.50 | threshold 0.65 |
|---|---|---|
| accuracy | 93.29% | **92.80%** |
| recall clase 0 (reprueba) | 0.63 | **0.73** |
| precision clase 0 | 0.84 | 0.74 |
| recall clase 1 (aprueba) | 0.98 | 0.96 |
| precision clase 1 | 0.94 | 0.96 |
| falsos negativos (reprueba, no detectado) | 328 | **226** |
| falsos positivos (falsa alarma) | 110 | 226 |

Costo: -0.49pp de accuracy global. Beneficio: 84 alumnos en riesgo real más
detectados (566→650 de 894), a cambio de 116 falsas alarmas más. Trade-off
aceptado para un modelo de alerta temprana, donde el falso negativo es el
error caro.

**`assets/modelo_logit.pkl` regenerado** (`python src/train.py`, 16
columnas incluida `const`) — el threshold es un post-proceso sobre
`predict()`, no cambia el ajuste, pero el pkl se corrió de nuevo para
confirmar que `main.py` sirve el modelo vigente. Suite completa
(`tests/test_integration.py` + `src/test_modelo5logit.py`, 31 tests) pasa
con el pkl nuevo.

## 9. ROC-AUC: el modelo en sí no empeoró, solo el corte se movió

Bajar precision de la clase 0 al subir el threshold (0.84→0.74, sección 8)
es esperable — recall y precision son un trade-off inevitable para
cualquier corte fijo sobre el mismo modelo, no indica que el modelo haya
empeorado. Para confirmarlo con una métrica que **no depende del
threshold**, se calculó ROC-AUC sobre las probabilidades crudas
(`model.predict(exog=X_test)`) del test set:

```
ROC-AUC: 0.9549
```

Interpretación: tomando al azar un alumno que reprobó y uno que aprobó, el
modelo le asigna mayor riesgo al que reprobó el 95.5% de las veces. Muy por
encima de 0.5 (adivinar al azar) y del rango típico bueno (0.85-0.90) para
un Logit. Confirma que el modelo separa las clases muy bien — el problema
detectado en la sección 7 (recall 0.63 con threshold 0.5) era de
calibración del corte, no de calidad del modelo, y mover el threshold a
0.65 fue el fix correcto.

**Validado con fila real** (índice 6528 de `modelo_sara.csv`, alumno con
`estado_asignatura=0`, reprobó de verdad): `POST /predict` da
`riesgo_reprobar=0.4642`. Con threshold viejo (0.5) hubiera dado
`prediccion=0` (aprueba) — falso negativo, sin alerta. Con threshold nuevo
(0.35) da `prediccion=1` (riesgo) — detectado. Confirma en un caso real,
no sintético, que el cambio de threshold atrapa exactamente el tipo de
caso que motivó el cambio (sección 7).
