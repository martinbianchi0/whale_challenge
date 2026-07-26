# 🐋 ¿Sirven los datos sintéticos para clasificar cantos de ballenas? Un resultado negativo, medido

Estudio controlado sobre si el **data augmentation con modelos generativos** mejora un clasificador de audio en la tarea de distinguir cantos de ballena de ruido de fondo.

**La respuesta que encontramos fue que no.** Ninguna de las tres familias generativas entrenadas —VAE, Adversarial Autoencoder y GAN— mejoró al clasificador entrenado solo con datos reales. Ese es el resultado del trabajo, y está reportado como tal.

Trabajo final de **Aprendizaje Automático y Aprendizaje Profundo (I302)** — Universidad de San Andrés, julio 2025. Por **Martín Bianchi** y **Federico Gutman**.

---

## Resultados

Las seis variantes se evalúan **sobre el mismo conjunto de validación**, con early stopping por *validation loss*. La única diferencia entre ellas es con qué datos se entrenó el clasificador (una CNN sobre espectrogramas, idéntica en todos los casos).

| Entrenamiento del clasificador | Accuracy | F1 | AUC |
|---|---|---|---|
| **Solo datos reales (baseline)** | **92.47 %** | **0.8471** | **0.9750** |
| Reales + augmentation clásica de audio | 92.47 % | 0.8411 | 0.9739 |
| Reales + sintéticos de **VAE** | 92.52 % | 0.8384 | 0.9751 |
| Reales + sintéticos de **GAN** | 92.23 % | 0.8348 | 0.9740 |
| Reales + sintéticos de **AAE** | 92.32 % | 0.8327 | 0.9736 |
| Reales, con ponderación de clases | 91.07 % | 0.8288 | 0.9741 |

**Cómo leerlo.** El baseline tiene el mejor F1 de los seis. El AUC se mueve en la cuarta cifra decimal (0.9736 a 0.9751), o sea dentro del ruido. El VAE gana 0.05 puntos de accuracy y **pierde** 0.9 puntos de F1. **No hay una variante generativa que mejore al baseline de forma consistente en las tres métricas** — y con un problema desbalanceado, el F1 es la métrica que importa.

La ponderación de clases, que era la intervención "obvia" para el desbalance, también empeoró el resultado.

**Resultado secundario:** los clasificadores sobre features tabulares quedan claramente por debajo de la CNN sobre espectrogramas — Random Forest 0.6863 de F1 (AUC 0.7783) y Gradient Boosting 0.7081 (AUC 0.7980), contra 0.8471 de la CNN. La representación importó más que el modelo.

## Por qué el resultado negativo es el aporte

Los generativos **sí aprendieron** a producir espectrogramas: el espacio latente se visualiza, las muestras sintéticas se comparan contra las reales por reducción de dimensionalidad, y son plausibles. Lo que no ocurrió es que **agregar esas muestras al entrenamiento aporte información que el clasificador no tuviera ya**. Es una distinción que se pierde si uno solo mira si el número final subió.

El informe (`BIANCHI_GUTMAN_INFORME_PF.pdf`) tiene el detalle: arquitecturas, espacios latentes, matrices de confusión y curvas de aprendizaje.

## Contenido

| Archivo / carpeta | Qué es |
|---|---|
| `model_training.ipynb` | Análisis exploratorio, baselines (MLP, Random Forest, Gradient Boosting) y la CNN sobre espectrogramas |
| `model_testing.ipynb` | La comparación controlada de las seis variantes de arriba |
| `src/models/generative.py` | VAE, Adversarial Autoencoder y GAN |
| `src/models/classification.py` | Clasificadores |
| `src/preprocessing.py`, `src/metrics.py`, `src/display.py` | Señal a espectrograma, métricas y visualización |
| `saved_models/` | Pesos de los generativos entrenados |
| `BIANCHI_GUTMAN_INFORME_PF.pdf` | Informe completo |
| `whale_detector/` | **Código de terceros**, no nuestro: la solución en PyTorch de Tarin Ziyaee al Kaggle Whale Detection Challenge (MIT, 2017), conservada como referencia de comparación |

## Alcance

Es coursework. Los datos vienen del Kaggle Whale Detection Challenge, la evaluación es sobre un conjunto de validación —no hay test set retenido— y las diferencias entre variantes son chicas, así que la conclusión correcta es "no se observó mejora", no "los generativos no sirven para esto".
