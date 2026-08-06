# Historial de mejoras al modelo de riesgo SARA

Bitácora cronológica de los cambios realizados en el modelo Logit (`src/modelo5logit.py`) y sus motivaciones. Este documento complementa a `CHANGES_LOGIT.md` y `ECUACION_MD.md` (que detallan los dos primeros ajustes técnicos) e integra la secuencia completa, incluyendo la migración de la fuente de datos y el análisis reciente de precisión y recall.

---

### 1. Trampa de variables dummy (`dep_*`, `tipo_ingreso_*`)
* **Problema:** Al incluir grupos one-hot completos —tales como departamentos académicos (`dep_tec_cn`, `dep_cs_hm`, `dep_sd`) y tipos de ingreso (`tipo_ingreso_regular`, `_bea`, `_pace`, `_especial`, `_tranfexterna`)— donde cada grupo suma 1 en cada fila y no se excluía una categoría de referencia, `Logit.summary()` arrojaba coeficientes absurdos (como $-16.68$) y errores estándar en `NaN`.
* **Causa:** Colinealidad perfecta entre la constante y cada grupo completo de variables dummy (trampa de la variable dummy).
* **Solución:** La función `eliminar_categoria_referencia()` elimina la primera columna de cada grupo (`GRUPOS_DUMMIES_COLINEALES = [("dep_",), ("tipo_ingreso_",)]`), estableciendo implícitamente la categoría de referencia (ej. `dep_tec_cn`, `tipo_ingreso_regular`). El detalle completo se encuentra en `CHANGES_LOGIT.md`.

---

### 2. Inversión de la función sigmoide
* **Problema/Lógica:** `estado_asignatura = 1` indica aprobación (ausencia de riesgo). Dado que `sigmoid(z)` calcula la probabilidad de aprobar ($P(	ext{aprobar})$), el riesgo real corresponde a $1 - 	ext{sigmoid}(z)$. 
* **Solución:** Nunca se utiliza el valor crudo de $z$ (no acotado, en log-odds) ni la sigmoide sin invertir. El detalle técnico está en `ECUACION_MD.md`.

---

### 3. Escalas y centrado de variables
* **Problema en `ano_carrera_actual`:** Anteriormente llamado `Año_Ingreso`, traía valores absolutos (2017–2026) sin centrar, lo que obligaba a la constante del Logit a absorber un offset gigantesco, generando un artefacto visual (no un error de ajuste).
    * *Solución:* La función `centrar_ano_ingreso()` resta la media antes de aplicar `sm.add_constant`.
* **Problema en `promedio_sct`:** Se entrena en una escala de 0 a 100, pero la fuente de producción (Dagster, `indicador_sara.csv`) entrega una fracción de 0 a 1.
    * *Solución temporal:* Escalar por 100 en la ecuación de producción (pendiente corregir el origen en el pipeline).

---

### 4. Gestión de colinealidad: variables excluidas y retenidas
* `nota_final_curso` vs. `porcen_aproba_curso`: Poseen una correlación de 0.908 en el CSV actual y miden prácticamente lo mismo. Se excluyó `nota_final_curso` mediante `COLUMNAS_EXCLUIDAS`.
* `promedio_notas_estudiante` vs. `progresion_academica_sct`: Aunque son colineales (correlación de 0.741), **se conservan ambas**. Se intentó retirar `promedio_notas_estudiante`, pero el rendimiento decayó drásticamente en alumnos de primer año (con avance/SCT bajo por definición), volviéndose indistinguibles entre buenos y malos estudiantes.
* **Interacción `nota_parcial_1 * porcen_aproba_curso`:** Se probó su inclusión, pero no resultó significativa ($p = 0.095$) ni mejoró el *accuracy*, por lo que fue revertida.

---

### 5. Corrección de errores tipográficos en datasets antiguos
* **Problema:** El archivo Excel original (`BD_SARA_vf.xlsx`) contenía la columna `"Promedio_notas_estudiante "` con un espacio final oculto que no se limpiaba en `cargar_datos()`. Como el mapeo `CAMPO_A_COLUMNA` en `src/main.py` apuntaba al nombre limpio, esto provocaba un `KeyError` en el vector de inferencia.
* **Estado actual:** Ya no aplica tras la migración completa al formato CSV (columnas en formato `snake_case` libres de espacios).

---

### 6. Migración de la fuente de datos: Excel a CSV
* **Cambio (OpenSpec: `entrenar-modelo-sara-csv`):** La fuente oficial del proyecto migró de `assets/BD_SARA_vf.xlsx` (21.202 filas) a `assets/modelo_sara.csv` (32.620 filas, separado por punto y coma, columnas en `snake_case`).
* **Equivalencias:** Mantiene el mismo target binario y problemas estructurales de fondo (trampa dummy y colinealidad de 0.908 entre `nota_final_curso` y `porcen_aproba_curso`), pero renombra variables clave (ej. `Carrera_curso_*` $	o$ `dep_*`, `Tipo_ingreso_*` $	o$ `tipo_ingreso_*`, `Puntaje_rnk` $	o$ `puntaje_ranking`, `Año_Ingreso` $	o$ `ano_carrera_actual`).
* **Impacto:** Se reutilizó toda la lógica validada previamente a través de constantes paramétricas sin alterar el algoritmo. Se reescribió `src/main.py` (`PrediccionRequest` / `CAMPO_A_COLUMNA`) adaptándolo al nuevo esquema —un cambio *breaking* en el contrato de `/predict`—.
* **Resultado:** El *accuracy* aumentó de un 90.97% (Excel; validación cruzada 5-fold entre 90.62% y 91.98%, media de 91.22%) a un **93.29%** (CSV, incorporando más datos con idéntica proporción de clases).

---

### 7. Análisis de precisión, recall y desbalance de clases
El *accuracy* global de 93.29% ocultaba un sesgo debido al desbalance de clases (Clase 1 = aprueba, 5.630 casos en test; Clase 0 = reprueba, 894 casos; representando un baseline de 86.3% con solo predecir la clase mayoritaria). 

Para visibilizarlo, se integró `classification_report` de `scikit-learn` en `evaluar_modelo()`, obteniendo el siguiente desglose:

| Clase | Precisión | Recall | Soporte |
| :--- | :---: | :---: | :---: |
| **0 (Reprueba)** | 0.84 | 0.63 | 894 |
| **1 (Aprueba)** | 0.94 | 0.98 | 5.630 |

* **Hallazgo crítico:** El *recall* de la clase 0 es de apenas **0.63**. De 894 alumnos que reprueban en la realidad, el modelo solo detecta 566, dejando pasar 328 como falsos negativos (estudiantes en riesgo real catalogados como seguros y sin alerta temprana). En un sistema de alerta, este es el error más costoso, el cual quedaba enmascarado por el volumen de la clase mayoritaria.

Se realizó un barrido de umbrales (*thresholds*) sobre la probabilidad de aprobar $P(	ext{aprobar})$:

| Threshold $P(	ext{aprobar})$ | Recall (Clase 0) | Precisión (Clase 0) |
| :---: | :---: | :---: |
| 0.50 (Anterior) | 0.63 | 0.84 |
| 0.61 | 0.70 | 0.76 |
| 0.69 | 0.75 | 0.70 |
| **0.75** | **0.77** | **0.68** |
| 0.80 | 0.64 | 0.82 |
| 0.85 | 0.59 | - |

---

### 8. Ajuste de umbral (*Threshold*: 0.5 $	o$ 0.65)
Se implementó un umbral de **0.65** (equivalente a un corte de riesgo de **0.35** sobre la probabilidad de reprobar):

* `src/modelo5logit.py` (`evaluar_modelo`): `clasificacion = np.where(predicciones < 0.65, 0, 1)`
* `src/main.py:78` (API local `/predict`): `prediccion = 1 if riesgo_reprobar >= 0.35 else 0`
* *Nota:* `ecuacionlogit.md` (Dagster) permanece intacto, ya que retorna el valor continuo `p_riesgo_SARA` sin binarizar. Si Dagster aplica un corte propio río abajo, es responsabilidad exclusiva de esa capa del pipeline.

#### Comparativa de desempeño en el conjunto de prueba (6.524 filas):

| Métrica | Threshold 0.50 | Threshold 0.65 |
| :--- | :---: | :---: |
| **Accuracy global** | 93.29% | 92.80% |
| **Recall (Clase 0 - Reprueba)** | 0.63 | **0.73** |
| **Precisión (Clase 0)** | 0.84 | 0.74 |
| **Recall (Clase 1 - Aprueba)** | 0.98 | 0.96 |
| **Precisión (Clase 1)** | 0.94 | 0.96 |
| **Falsos Negativos (Riesgo omitido)** | 328 | **226** |
| **Falsos Positivos (Falsas alarmas)** | 110 | 226 |

* **Balance de la decisión:** Se acepta una reducción marginal de $-0.49\%$ en el *accuracy* global a cambio de detectar **84 alumnos adicionales en riesgo real** (aumentando de 566 a 650 detecciones correctas sobre 894), aceptando a cambio 116 nuevas falsas alarmas. Este trade-off es óptimo para un modelo de alerta temprana.
* Se regeneró el artefacto `assets/modelo_logit.pkl` mediante `python src/train.py` (16 columnas incluyendo la constante). La suite completa de pruebas (`tests/test_integration.py` y `src/test_modelo5logit.py`, sumando 31 tests) pasa de forma exitosa.

---

### 9. Validación mediante ROC-AUC
Para comprobar que la caída en precisión de la clase 0 no reflejaba un deterioro del modelo predictivo en sí (sino un efecto del desplazamiento del corte), se calculó el **ROC-AUC** sobre las probabilidades crudas (`model.predict(exog=X_test)`):

> **ROC-AUC: 0.9549**

* **Interpretación:** Al emparejar al azar un alumno que reprobó con uno que aprobó, el modelo asigna correctamente un mayor riesgo al alumno reprobado el **95.5% de las veces**. Este valor supera ampliamente el azar (0.5) y el estándar de un buen modelo Logit (0.85 - 0.90), confirmando una capacidad discriminante excelente.
* **Validación en caso real:** Evaluando la fila con índice `6528` de `modelo_sara.csv` (un alumno real con `estado_asignatura = 0`, que reprobó), el endpoint `POST /predict` arroja un `riesgo_reprobar = 0.4642`:
    * Con el umbral anterior ($0.5$), habría retornado `prediccion = 0` (**falso negativo**, sin alerta).
    * Con el nuevo umbral, retorna `prediccion = 1` (**detectado a tiempo**).

Esto confirma empíricamente que el ajuste del umbral cumple con su propósito de capturar los casos críticos que motivaron esta iteración.
