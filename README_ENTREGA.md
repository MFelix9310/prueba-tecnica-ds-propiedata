# Entrega — Félix Ruiz

Branch: `prueba-tecnica/ruiz`. Todo el trabajo vive adentro de los notebooks; cada uno cierra con un resumen de las decisiones que tomé y por qué.

## Cómo correr

```bash
git clone https://github.com/MFelix9310/prueba-tecnica-ds-propiedata.git
cd prueba-tecnica-ds-propiedata
git checkout prueba-tecnica/ruiz
python -m venv .venv
.venv/Scripts/activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/01_limpieza.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_modelado.ipynb
```

Para ver los runs de MLflow después:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Y abrir `http://localhost:5000`.

## Qué hay en cada cosa

- `notebooks/00_eda.ipynb` — exploración previa de los 3 CSVs. No es entregable formal pero queda por si te interesa el camino.
- `notebooks/01_limpieza.ipynb` — Tarea 1. Limpieza por plataforma, schema target, unificación. Resumen al final.
- `notebooks/02_modelado.ipynb` — Tarea 2. Tres modelos comparados (Ridge, HistGradientBoosting, XGBoost), MLflow tracking, análisis de errores, resumen al final.
- `output/dataset_unificado.parquet` — salida de la Tarea 1.
- `PROPUESTA.md` — Parte B. Las 4 áreas restantes con priorización a un mes.

## Resultados

HistGradientBoosting ganó en test (split temporal con cutoff 2025-05-15): R² = 0.857, RMSE ≈ 256k, MAE ≈ 167k, MAPE ≈ 17.96%. XGBoost quedó en R² = 0.815, Ridge sirvió de piso (R² = 0.755). El detalle vive en `02_modelado.ipynb`.

## Lo que no llegué a hacer

- Matching de inmuebles únicos. La consigna lo deja para la propuesta escrita; está en la sección 4.1 de `PROPUESTA.md`.
- Tipo de cambio mensual real para los precios en USD. Usé constante = 1100, que sale del propio dataset (mediana USD 735 × 1100 ≈ mediana general en pesos).
- Tuneo fino de hiperparámetros. Los defaults razonables ya dan R² 0.857 y la rúbrica privilegia criterio sobre décimas.
- Lags espaciales con vecinos cercanos. Es la feature que más mueve la aguja en alquileres pero depende de tener el matching resuelto, sino arma leakage entre listings del mismo cluster.
- Análisis con SHAP o partial dependence plots. Quedó en feature importance via permutación más residuos.
