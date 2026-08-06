# Qué es `ecuacionlogit.md` y por qué no se puede pasar de OLS a Logit "solo cambiando los coeficientes"

## Origen de `ecuacionlogit.md`

`ecuacionlogit.md` (raíz del repo) es la combinación lineal
(`z_score = -11.9771 + ... `) que usa el pipeline Dagster (otro proyecto,
fuera de este repo) para calcular el riesgo de los estudiantes. Se arma
copiando los coeficientes de `assets/coeficientes_modelo5logit.csv`, salida
de entrenar `modelo5logit.py` contra `assets/modelo_sara.csv`.

**No alcanza con reemplazar coeficientes y dejar la fórmula igual** al pasar
de un modelo OLS a uno Logit — por eso este documento existe.

## Por qué eso no es correcto

En OLS, `y = z` directamente: la combinación lineal *es* la predicción, y por
eso puede diseñarse para que ya caiga entre 0 y 1 (o truncarse a mano).

En Logit, `z` (el `z_score` de `ecuacionlogit.md`) es el **log-odds**, no la
probabilidad. `z` no está acotado — puede ser cualquier número real, positivo
o negativo, de cualquier magnitud. Para obtener una probabilidad hay que
aplicar la función logística (sigmoide) sobre `z`:

```
p_aprobar = 1 / (1 + exp(-z))
```

Si en Dagster se toman los coeficientes del Logit pero se sigue usando `z`
como si fuera la probabilidad final (como se hacía con OLS), el resultado no
es una probabilidad válida: puede salir mayor a 1, negativo, o simplemente
sin la escala que se necesita para interpretarlo.

## Ojo: sigmoide sola tampoco es el riesgo — hay que invertirla

Aplicar la sigmoide no es suficiente por sí sola: `estado_asignatura=1`
significa **aprobó**, no riesgo (confirmado contra el esquema de datos). Por
lo tanto `sigmoid(z)` da `P(aprobar)`, y el riesgo real es su complemento:

```
p_aprobar_SARA = 1 / (1 + exp(-z_score))
p_riesgo_SARA  = 1 - p_aprobar_SARA
```

Usar `sigmoid(z)` directo como "riesgo" sin invertir marcaría como alto
riesgo a los alumnos con más probabilidad de aprobar — exactamente al revés.
`ecuacionlogit.md` ya hace esta inversión correctamente. `modelo5logit.py`
llega al mismo resultado indirectamente: `evaluar_modelo()` llama a
`model.predict(exog=X_test)` (que da `P(aprobar)`, sigmoide de statsmodels
por defecto) y compara contra el target `estado_asignatura` tal cual, sin
necesitar invertir porque no reporta un "riesgo" explícito — es Dagster,
consumiendo esos mismos coeficientes para producir `p_riesgo_SARA`, quien
necesita el paso de inversión.

## Centrado de `ano_carrera_actual`

`ano_carrera_actual` (antes `Año_Ingreso` en el Excel) trae valores tipo
2017-2026 sin centrar. El Logit necesita compensar ese offset grande
metiéndolo en la constante; sin centrar, el const sale con una magnitud
mucho mayor que el resto de los coeficientes (artefacto de escala, no un
error de ajuste ni de datos).

`modelo5logit.py` corrige esto con `centrar_ano_ingreso(X)`, que resta la
media de `ano_carrera_actual` antes de `sm.add_constant`. Con
`modelo_sara.csv`, esa media es **2021.02** y el const resultante es
**-11.9771** (`assets/coeficientes_modelo5logit.csv`).

**Importante para Dagster:** como se centra `ano_carrera_actual` en el
entrenamiento, hay que centrarlo con la misma media (2021.02) al calcular el
riesgo en producción — `ecuacionlogit.md` ya lo hace
(`v("ano_carrera_actual") - 2021.02`) — y siempre aplicar sigmoide + inversión
sobre `z`, nunca usar `z` directo como riesgo. Si se reentrena el modelo con
datos nuevos, esta media cambia y hay que actualizarla en `ecuacionlogit.md`
junto con el resto de los coeficientes.

## Estado vigente

`ecuacionlogit.md` ya está sincronizado con
`assets/coeficientes_modelo5logit.csv` (mismo const, mismos coeficientes,
misma media de centrado). El detalle de por qué cambiaron los nombres de
columna (`Carrera_curso_*`→`dep_*`, `Año_Ingreso`→`ano_carrera_actual`, etc.)
y del ajuste de threshold (0.5→0.65 sobre `p_aprobar`, 0.35 sobre
`p_riesgo_SARA`) está en [MEJORAS_MODELO.md](MEJORAS_MODELO.md), secciones 6
a 9 — no se duplica acá para no quedar desactualizado si se vuelve a
reentrenar.
