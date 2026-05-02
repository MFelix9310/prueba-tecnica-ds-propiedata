# Propuesta — los 4 frentes restantes

Después de hacer la limpieza y el modelado base, dejo acá cómo encararía las cuatro áreas que faltan. Cada una con diagnóstico corto, propuesta concreta, esfuerzo y riesgos. Al final, qué priorizaría si tuviera un mes.

---

## 4.1 Matching de inmuebles únicos

Hoy las tres plataformas describen el mismo universo de inmuebles pero no hay id común. Con la unificación trivial (concat) la tabla analítica queda con duplicados: el mismo depto puede aparecer 3 veces si lo publicaron en A, B y C, y varias veces dentro de la misma plataforma. Eso infla el dataset, sobrerrepresenta los barrios donde las tres plataformas pisan más fuerte y, lo más feo, rompe los splits temporales: la misma propiedad termina en train y en test.

La propuesta es un pipeline de tres etapas. Primero **blocking** para no hacer comparación cuadrática: agrupo por barrio + bucket de precio (±15%) + bucket de m² (±10%). Eso reduce el espacio de comparación de O(n²) a algo manejable en CPU. Después **scoring** combinando tres señales: distancia haversine entre coordenadas cuando ambas las traen (la más fuerte), similitud coseno entre embeddings de la descripción (un sentence-transformer multilingüe alcanza para 11k registros; en mi tesis usé BGE-large 1024d para 254M registros con PySpark, acá no hace falta tanto), y match aproximado en estructurales (ambientes, dorm, baños, m²) con peso por feature. La salida es un score combinado por par. Por último, **threshold tuning** sobre un set anotado a mano de 300-500 pares — se anota en una tarde con alguien de producto. Curva PR sobre el score y elegir el punto que privilegia precision (>0.95) sobre recall: un FP arma un cluster espurio que después promedia precios distintos y rompe la integridad; un FN solo deja un duplicado más, vivible.

Esfuerzo: 2-3 semanas de implementación. La parte cara es el set anotado y el ajuste fino del threshold, no el código.

Riesgos: falsos positivos en zonas con torres (Puerto Madero, Belgrano R) donde varias unidades distintas tienen mismo precio y mismo m² — el threshold tiene que ser muy estricto ahí. Falsos negativos cuando dos publicaciones del mismo inmueble tienen descripciones muy distintas; ahí el blocking estructural compensa la divergencia textual. Y como no hay id estable entre snapshots, hay que repetir el matching cada batch o persistir cluster_id y vincular nuevos listings al cluster más cercano.

---

## 4.2 Arquitectura de tablas

El docx dice textual que "hay desorden entre el output del scraping y el dataset final de entrenamiento". Sin separación de capas, cada cambio en feature engineering invalida el dataset histórico y no hay forma de reproducir un modelo de hace tres meses.

La propuesta es arquitectura medallón clásica con cuatro capas. **Bronze**: raw del scraping, particionado por (fecha_scrape, plataforma), append-only, snapshot diario. Sirve de auditoría y permite rebuilds completos. **Silver**: schema unificado — lo que armé en la Tarea 1. Una fila por listing-snapshot, sin matching aún, idempotente respecto a Bronze. **Gold**: dataset analítico con matching aplicado (ver 4.1), una fila por inmueble único × snapshot, features derivadas (distancia centro, lags espaciales). Esta es la entrada al training. **Serving**: vista materializada para la API, solo el último snapshot por inmueble y el feature set congelado del modelo en producción.

Para el stack arrancaría con DuckDB + PyArrow + Parquet en disco. En mi tesis hice algo parecido sobre 254M registros y 1.2TB, así que para Propiedata (11k listings, va a crecer pero está lejos de TB) sobra. Si el volumen escala, migrar a Iceberg o Delta sobre S3 sin tocar el código de transformación porque el formato Parquet se mantiene compatible.

Versionado: snapshot diario en Bronze, append-only en Silver y **table versioning** (Iceberg/Delta) en Gold. Eso permite que cada run de entrenamiento apunte a un snapshot determinístico de Gold y se pueda reproducir el modelo seis meses después.

Esfuerzo: 1-2 semanas para el MVP sin Iceberg. Otra semana si lo querés con table versioning desde el principio.

Riesgos: si el matching de 4.1 cambia de versión, Gold se invalida y hay que rebuildear. Mitigación: registrar la versión del matcher como metadata de cada snapshot de Gold. Costo de almacenamiento si Bronze se mantiene mucho — política simple de 90 días en hot storage y el resto se compacta o archiva. Y al principio sin Airflow ni DBT — un Python con timestamps alcanza para empezar; cuando crezca el equipo, agregar DBT para Silver→Gold y Airflow para schedule.

---

## 4.3 Modelado avanzado para R² ≥ 0.8

En la prueba técnica ya alcancé R²=0.86 en test con HistGradientBoosting, así que el 0.8 no es el techo: es el piso. Pero el RMSE absoluto sigue en ~250k ARS sobre una mediana de 810k, que es bastante. Y donde más se equivoca el modelo (cuartil bajo de precio con MAPE 32%, San Telmo con 40%) tiene techo claro.

Priorizo por impacto vs costo en tres empujones.

**Primero, alto impacto y bajo costo.** Target en log para amortiguar la cola de precio y ponderar mejor los errores relativos — espero 1-2 puntos de R² y bajada del MAPE. Segmentación por tipo de propiedad (depto, PH y casa son economías distintas, un modelo único promedia estructuras). Y la feature que más mueve la aguja en datos inmobiliarios: precio mediano de los k=5 vecinos más cercanos en los últimos 30 días. Esto último depende de tener el matching resuelto, sino hay leakage entre vecinos del mismo cluster.

**Segundo, impacto y costo medio.** Features espaciales adicionales — distancia a estaciones de subte, densidad comercial por barrio, índice socioeconómico INDEC. Hay datos públicos en BA Data que son gratis y aportan señal. Y segmentación por zona (premium vs resto): modelos separados para Palermo + Recoleta + Belgrano frente al resto suelen capturar mejor la varianza local que un modelo único.

**Tercero, retorno marginal.** Stacking HGB + XGB + Ridge regularizado con un meta-learner lineal. Cuando la familia ya está cerca del techo de los datos, los ensembles dan poco. Tuneo con Optuna sobre un espacio amplio — vale después de las features, no antes. El modelo no falla por hiperparámetros, falla por features.

Esfuerzo: empujón 1 una semana, empujón 2 dos semanas (depende de la velocidad para integrar fuentes externas), empujón 3 una semana extra.

Riesgos: lags espaciales con leakage si los vecinos cercanos pertenecen al mismo cluster de matching; restringir el k-NN a inmuebles distintos. Modelos por segmento requieren muestra suficiente — PH (~530 listings) y casa (~270) post-cleaning están en el límite, mejor un solo modelo con `tipo_propiedad` como feature pero pesando inverso a la frecuencia.

---

## 4.4 MLOps / CI-CD

El docx confirma que hoy no hay tracking sistemático ni despliegue automatizado. Sin tracking no hay forma de saber qué hiperparámetros generaron el modelo en producción. Sin monitoreo, una distribución cambiada (ej: el scraper rompe el porcentaje de USD vs ARS y nadie se entera) degrada el modelo en silencio.

**Stack mínimo viable** — esto es lo que armé en StrucMind.ai con resultados directos en producción (90% menos incidencias):

- MLflow para tracking + registry. Un experimento por modelo, runs versionados con dataset hash y git sha. El modelo en producción es siempre un `champion` con tag explícito en el registry.
- GitHub Actions para CI: tests unitarios sobre el código de features y entrenamiento, linting (ruff), validación de schema con Pandera o Great Expectations sobre Bronze y Silver.
- Docker para reproducibilidad. Imagen del servicio de inferencia y otra de entrenamiento. Mismas versiones de paquetes en dev/staging/prod.
- Evidently para drift básico. Distribución de features clave y target. Alerta si algún feature crítico se desvía más de 2 sigma respecto al baseline del champion.

**Stack ideal**, cuando el volumen lo justifique:

- Feature store (Feast). Hoy no es crítico porque el feature engineering es leve, pero cuando se sumen los lags espaciales de 4.3, tener consistencia entre el cómputo offline y la API es importante.
- A/B testing en producción. Ramear 5% del tráfico al challenger y comparar métricas de negocio (no de modelo). Si el challenger gana en negocio, promover.
- Alertas con thresholds dinámicos. En vez de hardcodear sigma, usar la variabilidad histórica del feature como baseline rolling.
- Rollback automático. Si la métrica online del champion cae más de un umbral en una ventana, volver al modelo anterior sin intervención humana.

Esfuerzo: 2 semanas para MVP. 4-6 semanas adicionales para el stack ideal, en gradual.

Riesgos: sobreingeniería temprana. El stack ideal solo vale con un equipo dedicado y varios modelos en producción. Para 1-2 modelos, el MVP alcanza. Drift detection con poco volumen va a ser ruidoso al principio — calibrar los thresholds después de acumular un par de meses de datos online.

---

## priorización si tuviera un mes

- **Semana 1**: matching de inmuebles únicos (4.1). Es lo que más impacta en la calidad del dataset y bloquea todo lo demás. Sin esto, el resto se construye sobre arena.
- **Semana 2**: arquitectura de tablas (4.2). Bronze → Silver → Gold. Sin separación de capas, el dataset que produce el matching es otro parche más, no una capa estable. Es foundation.
- **Semana 3**: modelado avanzado (4.3), primer empujón. Target log + segmentación por tipo + lags espaciales sobre los clusters del matching. Esos son los puntos de R² más fáciles de ganar y los que mejor se ven en métricas de negocio.
- **Semana 4**: MLOps mínimo viable (4.4). Tracking + Docker + alertas básicas. Dejar el modelo reproducible y monitoreado.

El orden importa. Si invertís 4.4 antes que 4.1, terminás automatizando un pipeline que entrena sobre data con duplicados — modelo malo, pero malo en forma reproducible. La foundation (matching + arquitectura) va primero, las mejoras (modelado + MLOps) construyen encima.
