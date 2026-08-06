
def _calcular_semestre(ano_cursados: Optional[int], periodo: Optional[str]) -> int:
    # semestre_academico en modelo_sara.csv solo tiene valores 1 y 2 (semestre
    # de dictado del ramo, no semestre acumulado de carrera del alumno). El
    # coeficiente del modelo (-0.5941) se ajusto para ese rango [1,2]; usar
    # ano_cursados para extrapolar (ej. semestre 11) le da al modelo un input
    # fuera de lo que vio en el entrenamiento. Requiere reentrenar el modelo
    # con una variable de semestre de carrera si se quiere ese comportamiento.
    if periodo is None:
        return 0
    periodo_str = str(periodo)
    if periodo_str.endswith(".1") or periodo_str.endswith("-1"):
        return 1
    if periodo_str.endswith(".2") or periodo_str.endswith("-2"):
        return 2
    return 0


def _calcular_ado(context,fila: Dict[str, Any]) -> float:
    def v(key):
        val = fila.get(key)
        return float(val) if val is not None else 0.0

    semestre = _calcular_semestre(fila.get("ano_cursados"), fila.get("periodo"))
    ti = _one_hot_via_ingreso(fila.get("via_ingreso"))

    z_score = (
           -11.9771
         + -0.0269    * (v("ano_carrera_actual") - 2021.02)      # centrado: usar la MISMA media (2021.02) que uso el entrenamiento, no el año crudo
         + 0            * v("dep_tec_cn")                        # categoria de referencia (ver nota abajo)
         - 1.1298     * v("dep_cs_hm")
         - 1.2392     * v("dep_sd")
         + 0.9514     * v("nota")
         + 0.0134     * v("total_curso")
         + 0.0492     * v("tasa_aprobacion")
         - 0.5941     * semestre
         - 0.0002     * v("puntaje_ranking_homologado")
         + 0            * ti["tipo_ingreso_regular"]              # categoria de referencia (ver nota abajo)
         - 0.5068     * ti["tipo_ingreso_bea"]
         - 0.2171     * ti["tipo_ingreso_pace"]
         + 0.0240     * ti["tipo_ingreso_especial"]
         - 0.1664     * ti["tipo_ingreso_tranfexterna"]
         + 1.2591     * v("promedio_total")
         + 0.0192     * (v("promedio_sct") * 100)                # entrenado en escala 0-100 (progresion_academica_sct), indicador_sara.csv trae fraccion 0-1: hay que escalar
         - 0.0626     * v("creditos")
    )
    context.log.info(f"[SARA EVAL] Alumno RUT: {v('rut')} | z_score calculado: {z_score} | promedio_sct: {v('promedio_sct')}")
    # riesgo=reprobar: clasifica 1 (riesgo) si z_score corresponde a prob_aprobar < 0.65,
    # es decir riesgo_reprobar >= 0.35 (ver modelo5logit.py evaluar_modelo + src/main.py,
    # threshold ajustado de 0.5 a 0.65/0.35 en MEJORAS_MODELO.md seccion 8). Este archivo
    # devuelve el riesgo continuo; si Dagster binariza rio abajo, usar 0.35 como corte
    # sobre p_riesgo_SARA, no 0.5.
    # LÓGICA DE CONTROL DE MÁRGENES SARA (IF / ELSE)
    # los suaviza y los manda a un riesgo bajo real y continuo (ej: -0.5000 -> 37% de riesgo)
    # z_score es el log-odds de Estado_Asignatura=1 (APROBAR, no reprobar: confirmado
    # contra BD_SARA_vf.xlsx), asi que sigmoid(z_score) da P(aprobar), no el riesgo.
    # El riesgo de reprobar es el complemento.
    p_aprobar_SARA = 1.0 / (1.0 + np.exp(-z_score))
    p_riesgo_SARA = 1.0 - p_aprobar_SARA

    # Toque final de seguridad para que la base de datos Postgres reciba valores limpios entre 0 y 1
    return p_riesgo_SARA

<!--
Nota importante (actualizado 2026-08-06, migracion Excel -> CSV):

Estos coeficientes salen de modelo5logit.py post-fix (eliminar_categoria_referencia
+ centrar_ano_ingreso), corrida sobre assets/modelo_sara.csv (antes BD_SARA_vf.xlsx).
Ver CHANGES_LOGIT.md, ECUACION_MD.md y MEJORAS_MODELO.md para el detalle completo
de por que cambio la ecuacion, incluida la migracion de fuente de datos y el
ajuste de threshold (seccion 8 de MEJORAS_MODELO.md).

0. **PENDIENTE DE CONFIRMAR CON EL PIPELINE DAGSTER (fuera de este repo):**
   el grupo de carrera cambio de 9 categorias ("Carrera_curso_*": agro, enf,
   icind, icinf, ifor, obs, psi, tsoc, multiple) a 3 categorias de
   DEPARTAMENTO academico ("dep_*": dep_tec_cn, dep_cs_hm, dep_sd) — no es
   el mismo concepto, es un cambio de granularidad distinto. Esta version
   asume que el `fila` dict que llega a `_calcular_ado` va a traer
   `dep_cs_hm` y `dep_sd` ya como 0/1 (mismo patron que antes con
   `carrera_curso_*`, leidos directo con `v()`, sin pasar por un helper
   one-hot). Si el pipeline Dagster todavia manda campos de carrera en vez
   de departamento, hay que agregar la logica de mapeo/one-hot ANTES de
   activar esta version — sino el modelo va a leer 0.0 en dep_cs_hm/dep_sd
   para todos los alumnos (default de `v()` cuando la key no esta), lo cual
   silenciosamente clasifica a todos como si fueran del departamento de
   referencia (dep_tec_cn). Mismo chequeo aplica a `_one_hot_via_ingreso`:
   debe devolver las keys nuevas (`tipo_ingreso_bea`, `_pace`, `_especial`,
   `_tranfexterna`) en vez de la vieja `tipo_ingreso_repostulacion` (esa
   categoria ya no existe en el esquema nuevo).

1. "dep_tec_cn" y "tipo_ingreso_regular" valen 0 porque son la categoria de
   referencia implicita (dummy variable trap corregida, ver CHANGES_LOGIT.md).
   No son coeficientes "faltantes": el optimizador no puede estimar un
   coeficiente propio para todas las categorias de un grupo one-hot
   completo junto a la constante (colinealidad perfecta) — hay que fijar
   una como base.

2. z (el resultado de esta formula) es log-odds, no el riesgo final. Para
   obtener el riesgo hay que aplicar la sigmoide:
       riesgo = 1 / (1 + exp(-z))
   No usar z directo como probabilidad.

3. Que "dep_tec_cn" y "regular" valgan 0 no significa que se ignoren o que
   no se consideren sus datos. Significa que son el comportamiento BASE del
   modelo: el "const" (-11.9771) ya ES el z de un estudiante del
   departamento tec_cn con ingreso regular y todo lo demas en su
   categoria/valor base. El resto de los departamentos y tipos de ingreso
   se leen como la diferencia respecto a esa base, no como valores
   independientes. Ejemplo: un estudiante de dep_cs_hm tiene
   z = const + (-1.1298) + resto; uno de dep_tec_cn tiene z = const + resto.
   El -1.1298 ES la comparacion tec_cn vs cs_hm, no un numero que "le falta"
   a cs_hm.

   Matematicamente da exactamente el mismo riesgo final que si "dep_tec_cn"
   y "regular" tuvieran coeficiente propio y el const fuera distinto (es la
   misma ecuacion escrita de otra forma) — de hecho asi estaba antes de
   corregir la trampa de variable dummy (coeficientes inestables, NaN en
   los errores estandar). Por eso se fija una categoria base por grupo.

   Por que "dep_tec_cn" quedo como base: no fue elegido por ningun criterio
   estadistico, es simplemente la primera columna "dep_*" en el orden de
   `modelo_sara.csv`. Si el orden de columnas del CSV cambiara, la base
   cambiaria de departamento, pero el riesgo calculado para cada estudiante
   seria identico.

4. Threshold de clasificacion (si Dagster binariza el riesgo continuo que
   devuelve esta funcion): usar 0.35 sobre `p_riesgo_SARA`, no 0.5 — ver
   punto 0 en MEJORAS_MODELO.md seccion 8 para el detalle del cambio y por
   que (recall de la clase "reprueba" subio de 0.63 a 0.73 con ese corte).
-->
