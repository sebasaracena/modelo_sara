from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
RUTA_DATOS = ASSETS_DIR / "modelo_sara.csv"
VARIABLE_DEPENDIENTE = "estado_asignatura"
RUTA_MATRIZ_CONFUSION = ASSETS_DIR / "confusion_matrix_modelo5logit.png"
RUTA_COEFICIENTES = ASSETS_DIR / "coeficientes_modelo5logit.csv"

# Grupos de columnas one-hot presentes en modelo_sara.csv: cada grupo suma 1
# por fila, por lo que sumado a la constante producen colinealidad perfecta
# (trampa de variable dummy) y coeficientes/errores estandar NaN en el Logit.
GRUPOS_DUMMIES_COLINEALES = [
    ("dep_",),
    ("tipo_ingreso_",),
]

# ano_carrera_actual no esta centrado (valores tipo 2017-2026): eso obliga a
# la constante del Logit a compensar ese offset y sale con un valor enorme
# sin que el modelo este mal. Centrar en la media evita ese artefacto
# visual sin cambiar ninguna prediccion.
COLUMNA_ANO_INGRESO = "ano_carrera_actual"

# nota_final_curso y porcen_aproba_curso miden lo mismo (dificultad del ramo:
# corr 0.908 en modelo_sara.csv), meterlas juntas desestabiliza sus
# coeficientes. Se deja porcen_aproba_curso por ser mas directa de
# interpretar.
#
# promedio_notas_estudiante y progresion_academica_sct tambien miden casi lo
# mismo en promedio (corr 0.741), pero NO se sacan: para alumnos recien
# ingresados (avance/SCT bajo por definicion, sin importar que tan buenas
# sean sus notas) promedio_notas_estudiante es la unica variable que permite
# distinguir un alumno nuevo bueno de uno malo. Sacarla infla el riesgo de
# todos los alumnos nuevos por igual. Se deja la colinealidad como
# limitacion conocida en vez de perder esa señal.
COLUMNAS_EXCLUIDAS = ["nota_final_curso"]


@dataclass
class ResultadoEvaluacion:
    accuracy: float
    clasificacion: np.ndarray
    matriz_confusion: pd.DataFrame
    reporte_clasificacion: str
    roc_auc: float


def cargar_datos(ruta=RUTA_DATOS):
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el archivo de datos: {ruta}")
    return pd.read_csv(ruta, sep=";")


def eliminar_categoria_referencia(X, grupos=GRUPOS_DUMMIES_COLINEALES):
    columnas_a_eliminar = []
    for prefijos_variantes in grupos:
        columnas_grupo = [
            c for c in X.columns if c.startswith(prefijos_variantes)
        ]
        if columnas_grupo:
            columnas_a_eliminar.append(columnas_grupo[0])
    return X.drop(columns=columnas_a_eliminar)


def centrar_ano_ingreso(X, columna=COLUMNA_ANO_INGRESO):
    if columna not in X.columns:
        return X
    X = X.copy()
    X[columna] = X[columna] - X[columna].mean()
    return X


def entrenar_modelo(datos, variable_independiente=VARIABLE_DEPENDIENTE):
    # Se probo agregar una interaccion nota_parcial_1 x porcen_aproba_curso
    # (hipotesis: nota buena deberia compensar mas en ramos dificiles) y se
    # revirtio: el termino no fue significativo y desestabilizo el
    # coeficiente propio de nota_parcial_1, sin mejorar el accuracy. Los
    # datos no respaldan esa interaccion lineal.
    variables_dependientes = [
        c for c in datos.columns
        if c != variable_independiente and c not in COLUMNAS_EXCLUIDAS
    ]
    X = eliminar_categoria_referencia(datos[variables_dependientes])
    X = centrar_ano_ingreso(X)
    X = sm.add_constant(X)
    y = datos[variable_independiente]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = sm.Logit(y_train, X_train).fit(disp=0)
    return model, X_train, X_test, y_train, y_test


def evaluar_modelo(model, X_test, y_test):
    predicciones = model.predict(exog=X_test)
    clasificacion = np.where(predicciones < 0.65, 0, 1)
    accuracy = accuracy_score(y_true=y_test, y_pred=clasificacion, normalize=True)
    matriz_confusion = pd.crosstab(
        y_test.to_numpy(),
        clasificacion,
        rownames=["Real"],
        colnames=["Prediccion"],
    )
    reporte_clasificacion = classification_report(y_true=y_test, y_pred=clasificacion)
    roc_auc = roc_auc_score(y_true=y_test, y_score=predicciones)
    return ResultadoEvaluacion(
        accuracy=accuracy,
        clasificacion=clasificacion,
        matriz_confusion=matriz_confusion,
        reporte_clasificacion=reporte_clasificacion,
        roc_auc=roc_auc,
    )


def obtener_coeficientes(model):
    return pd.DataFrame(
        {
            "variable": model.params.index,
            "coef": model.params.values,
            "pvalue": model.pvalues.values,
            "odds_ratio": np.exp(model.params.values),
        }
    )


def guardar_matriz_confusion(matriz_confusion, ruta=RUTA_MATRIZ_CONFUSION):
    ruta = Path(ruta)
    fig, ax = plt.subplots()
    sns.heatmap(matriz_confusion, annot=True, cmap="YlGnBu", fmt="g")
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title("Confusion matrix", y=1.1)
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


def main():
    datos = cargar_datos()

    print("Numero de observaciones por clase")
    print(datos[VARIABLE_DEPENDIENTE].value_counts())
    print()
    print("Porcentaje de observaciones por clase")
    print(100 * datos[VARIABLE_DEPENDIENTE].value_counts(normalize=True))

    model, X_train, X_test, y_train, y_test = entrenar_modelo(datos)
    print(model.summary())

    coeficientes = obtener_coeficientes(model)
    coeficientes.to_csv(RUTA_COEFICIENTES, index=False)
    print(f"\nCoeficientes guardados en: {RUTA_COEFICIENTES}")

    resultado = evaluar_modelo(model, X_test, y_test)
    print(f"\nEl accuracy de test es: {100 * resultado.accuracy}%")
    print(resultado.matriz_confusion)
    print("\nPrecision / recall / f1 por clase:")
    print(resultado.reporte_clasificacion)
    print(f"ROC-AUC (sobre probabilidades crudas de aprobar): {resultado.roc_auc:.4f}")
    print(
        "\nNota: umbral de clasificacion ajustado de 0.5 a 0.65 a proposito "
        "(ver MEJORAS_MODELO.md seccion 8). Prioriza recall de clase 0 "
        "(riesgo detectado) sobre accuracy global: con 0.5 el recall clase 0 "
        "era 0.63, con 0.65 sube a ~0.73 a cambio de -0.5pp de accuracy y mas "
        "falsas alarmas. Precision/recall de clase 0 ~0.73-0.74 es esperado, "
        "no una regresion."
    )

    ruta_imagen = guardar_matriz_confusion(resultado.matriz_confusion)
    print(f"Matriz de confusion guardada en: {ruta_imagen}")


if __name__ == "__main__":
    main()
