# Corrección: trampa de variable dummy en modelo5logit.py

## Qué pasaba

Al correr `modelo5logit.py` sin `eliminar_categoria_referencia()`, el
`Logit.summary()` muestra coeficientes absurdos y errores estándar `NaN` en
varias variables. El bug se descubrió originalmente contra `BD_SARA_vf.xlsx`
(columnas `Carrera_curso_*`/`Tipo_ingreso_*`); tras la migración a
`assets/modelo_sara.csv` (ver [MEJORAS_MODELO.md](MEJORAS_MODELO.md),
sección 6) el mismo mecanismo aplica sobre `dep_*`/`tipo_ingreso_*`:

```
dep_cs_hm                    -1.1298        nan        nan        nan
tipo_ingreso_especial        -16.6844   1.07e+06  -1.55e-05      1.000
```

(valores ilustrativos del mismo patrón; sin el fix, `dep_*` y
`tipo_ingreso_*` quedan con coeficientes inestables y `bse` en `NaN`).

## Por qué pasaba

En `modelo_sara.csv` las columnas `dep_*` (3 columnas: `dep_tec_cn`,
`dep_cs_hm`, `dep_sd`) y `tipo_ingreso_*` (5 columnas: `tipo_ingreso_regular`,
`_bea`, `_pace`, `_especial`, `_tranfexterna`) son codificaciones one-hot
completas: para cada fila, las columnas de cada grupo suman exactamente 1.
Ningún grupo tiene una categoría de referencia excluida.

Al agregarles una constante (`sm.add_constant`), la matriz de diseño queda con
colinealidad perfecta (rank-deficient): la constante es una combinación lineal
exacta de cada grupo de dummies. El optimizador de máxima verosimilitud del
Logit "converge" (`converged: True`) pero no puede separar el efecto de la
constante del de cada dummy, así que empuja algunos coeficientes a valores
extremos y el cálculo del error estándar (que depende de invertir la matriz
Hessiana) se vuelve indefinido → `NaN`.

Esto es la **trampa de la variable dummy** (dummy variable trap), un problema
clásico de codificación, no un problema de los datos ni de las clases estar
desbalanceadas.

## Qué se corrigió

En `modelo5logit.py` está `eliminar_categoria_referencia(X)`, que dropea la
primera columna de cada grupo one-hot listado en `GRUPOS_DUMMIES_COLINEALES
= [("dep_",), ("tipo_ingreso_",)]` antes de agregar la constante. Esa columna
eliminada pasa a ser la **categoría de referencia implícita** del grupo.

Con `modelo_sara.csv`, la referencia es `dep_tec_cn` y `tipo_ingreso_regular`
(son las primeras columnas de cada grupo en el orden del CSV — no un
criterio estadístico, ver nota en `ecuacionlogit.md`). El resto de las
dummies del grupo quedan en el modelo, interpretables respecto a esa
referencia.

`modelo5.py` (OLS) no se toca: statsmodels solo reporta un *warning* de
multicolinealidad ahí en vez de `NaN`, y el alcance es exclusivamente el
Logit.

## Impacto en los odds ratios (`coeficientes_modelo5logit.csv`)

- **Sin el fix:** los odds ratio de `dep_*` y `tipo_ingreso_*` no serían
  interpretables (`exp` de coeficientes extremos, con p-value `NaN` o
  `1.000`).
- **Con el fix (valores reales, `modelo_sara.csv`):** cada odds ratio de esas
  dummies se lee como "odds relativas a la categoría de referencia excluida
  del grupo", con p-values normales:
  - `dep_cs_hm`: OR ≈ 0.323, p < 0.001 → estudiantes de ese departamento
    tienen ~68% menos probabilidad relativa (odds) de aprobar frente al
    departamento de referencia (`dep_tec_cn`), controlando por el resto de
    variables.
  - `dep_sd`: OR ≈ 0.290, p < 0.001 → ~71% menos odds que `dep_tec_cn`.
  - Los `tipo_ingreso_*` (`_bea`, `_pace`, `_especial`, `_tranfexterna`) NO
    son estadísticamente significativos en el CSV actual (p entre 0.17 y
    0.74): a diferencia de la corrida anterior sobre el Excel, con estos
    datos no hay evidencia de que el tipo de ingreso mueva el riesgo una vez
    controladas las demás variables.
- El accuracy de test **no cambia** por este fix específico: solo hace que
  los coeficientes sean matemáticamente válidos e interpretables, no altera
  la capacidad predictiva del modelo. Para el número de accuracy vigente y
  su historial ver [MEJORAS_MODELO.md](MEJORAS_MODELO.md), secciones 6 a 9.

## Cómo se verificó (TDD)

El fix se desarrolló originalmente con TDD contra `src/test_modelo5logit.py`
(datos sintéticos con dos grupos one-hot completos, para reproducir el bug
sin depender del Excel real): tests que confirmaban que
`eliminar_categoria_referencia` dropea una columna por grupo detectado, no
toca columnas fuera de esos grupos, y que sin `NaN` en `model.bse` con datos
sintéticos que reproducen la trampa.

**Ese archivo de test ya no existe en el repo** (no sobrevivió a la
migración Excel → CSV, ver [MEJORAS_MODELO.md](MEJORAS_MODELO.md) sección 6).
La cobertura vigente sobre este comportamiento es indirecta: `tests/test_integration.py`
prueba el endpoint `/predict` end-to-end (que depende de que el `.pkl`
entrenado con `eliminar_categoria_referencia` esté bien formado), y correr
`python src/modelo5logit.py` contra `modelo_sara.csv` real confirma que el
`summary()` no muestra `NaN` ni coeficientes extremos. Si se vuelve a tocar
`eliminar_categoria_referencia`, conviene reponer un test unitario dedicado
antes de modificarla.
