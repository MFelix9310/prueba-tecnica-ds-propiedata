"""Tests basicos sobre el parquet unificado de la Tarea 1.

Validan invariantes que tienen que cumplirse despues de correr 01_limpieza.ipynb.
Para correrlos: pytest tests/  (desde la raiz del repo).
"""
from pathlib import Path
import pandas as pd
import pytest

PARQUET = Path(__file__).resolve().parent.parent / 'output' / 'dataset_unificado.parquet'


@pytest.fixture(scope='module')
def df():
    if not PARQUET.exists():
        pytest.skip(f'falta el parquet en {PARQUET}. Correr antes 01_limpieza.ipynb')
    return pd.read_parquet(PARQUET)


def test_tiene_filas_y_columnas_esperadas(df):
    assert len(df) > 9_000, 'esperaba al menos 9k filas post-limpieza'
    columnas_obligatorias = {
        'listing_id', 'plataforma', 'fecha_publicacion', 'tipo_propiedad',
        'barrio', 'lat', 'lng', 'precio_ars_mes', 'moneda_origen',
    }
    assert columnas_obligatorias.issubset(df.columns)


def test_precio_dentro_del_rango_admisible(df):
    # filtre 100k-10M en limpieza, ningun outlier debe sobrevivir
    assert df['precio_ars_mes'].min() >= 100_000
    assert df['precio_ars_mes'].max() <= 10_000_000


def test_no_hay_filas_sin_target(df):
    # las "Consultar" de C y los typos de A se descartan: el target nunca puede ser nulo
    assert df['precio_ars_mes'].notna().all()


def test_tipos_propiedad_normalizados(df):
    assert set(df['tipo_propiedad'].unique()) == {'depto', 'ph', 'casa'}


def test_listing_id_lleva_prefijo_de_plataforma(df):
    for plat in ['A', 'B', 'C']:
        sub = df[df['plataforma'] == plat]
        assert sub['listing_id'].str.startswith(f'{plat}-').all(), f'plataforma {plat} con id mal prefijado'


def test_flags_de_missing_son_booleanos(df):
    for col in ['flag_missing_expensas', 'flag_missing_antiguedad', 'flag_missing_m2_total']:
        assert df[col].dtype == bool, f'{col} deberia ser bool'
