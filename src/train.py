from pathlib import Path

import joblib

from modelo5logit import COLUMNA_ANO_INGRESO, cargar_datos, entrenar_modelo

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
RUTA_MODELO = ASSETS_DIR / "modelo_logit.pkl"


def entrenar_y_guardar(datos, ruta=RUTA_MODELO):
    model, X_train, X_test, y_train, y_test = entrenar_modelo(datos)
    media_ano_ingreso = float(datos[COLUMNA_ANO_INGRESO].mean())
    payload = {
        "model": model,
        "columnas_x": list(X_train.columns),
        "media_ano_ingreso": media_ano_ingreso,
    }
    joblib.dump(payload, ruta)
    return payload


def main():
    datos = cargar_datos()
    payload = entrenar_y_guardar(datos)
    print(f"Modelo guardado en: {RUTA_MODELO}")
    print(f"Columnas ({len(payload['columnas_x'])}): {payload['columnas_x']}")


if __name__ == "__main__":
    main()
