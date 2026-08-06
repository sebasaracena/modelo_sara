from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
RUTA_DATOS = ASSETS_DIR / "BD_SARA_vf.xlsx"
VARIABLE_DEPENDIENTE = "Estado_Asignatura"
RUTA_MATRIZ_CONFUSION = ASSETS_DIR / "confusion_matrix_modelo5.png"
RUTA_COEFICIENTES = ASSETS_DIR / "coeficientes_modelo5.csv"


@dataclass
class ResultadoEvaluacion:
    accuracy: float
    clasificacion: np.ndarray
    matriz_confusion: pd.DataFrame
    reporte_clasificacion: str


def cargar_datos(ruta=RUTA_DATOS):
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontro el archivo de datos: {ruta}")
    return pd.read_excel(ruta)


def entrenar_modelo(datos, variable_independiente=VARIABLE_DEPENDIENTE):
    variables_dependientes = [c for c in datos.columns if c != variable_independiente]
    X = sm.add_constant(datos[variables_dependientes])
    y = datos[variable_independiente]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = sm.OLS(y_train, X_train).fit()
    return model, X_train, X_test, y_train, y_test


def evaluar_modelo(model, X_test, y_test):
    predicciones = model.predict(exog=X_test)
    clasificacion = np.where(predicciones < 0.5, 0, 1)
    accuracy = accuracy_score(y_true=y_test, y_pred=clasificacion, normalize=True)
    matriz_confusion = pd.crosstab(
        y_test.to_numpy(),
        clasificacion,
        rownames=["Real"],
        colnames=["Prediccion"],
    )
    reporte_clasificacion = classification_report(y_true=y_test, y_pred=clasificacion)
    return ResultadoEvaluacion(
        accuracy=accuracy,
        clasificacion=clasificacion,
        matriz_confusion=matriz_confusion,
        reporte_clasificacion=reporte_clasificacion,
    )


def obtener_coeficientes(model):
    return pd.DataFrame(
        {"variable": model.params.index, "coef": model.params.values, "pvalue": model.pvalues.values}
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

    ruta_imagen = guardar_matriz_confusion(resultado.matriz_confusion)
    print(f"Matriz de confusion guardada en: {ruta_imagen}")


if __name__ == "__main__":
    main()
