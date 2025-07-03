# 🐋 Estudio de Data Augmentation en Audios de Ballenas con Modelos Generativos__<br/>


Este proyecto explora la clasificación de sonidos de ballenas utilizando modelos generativos y técnicas de data augmentation basadas en autoencoders y redes adversarias.

---

## 🚀 Objetivo general

Investigar si el uso de datos sintéticos generados por modelos generativos (AE, VAE, GAN) puede mejorar el desempeño de clasificadores de audio en la tarea de distinguir cantos de ballenas frente a ruido de fondo.

---

## 🧭 Roadmap del proyecto

El desarrollo se organiza en **etapas** con **checkpoints intermedios**, que permiten evaluar el progreso y la utilidad de cada técnica incorporada.

---

### 🔹 Etapa 0 — Análisis exploratorio
- Objetivo: Comprender la estructura del dataset.
- Acciones: Visualización de espectrogramas, inspección de clases, balance de datos, extracción de características de los espectrogramas y el audio, reducción de dimensionalidad.

---

### 🔹 Etapa 1 — Clasificador baseline
- Objetivo: Establecer un punto de partida con datos reales.
- Preprocesamiento: Conversión de la señal de audio cruda a espectrogramas.
- Modelos: MLP sin convoluciones, Random Forest, Gradient Boosting, MLP con convoluciones.
- Métricas: Accuracy, F1-score, matriz de confusión, AUC-ROC, curvas de aprendizaje.
- 📌 **Checkpoint 1**: Registrar la performance de datos reales y modelos simples.

---

### 🔹 Etapa 2 — Modelado generativo (VAE / AAE / GAN)
- Objetivo: Aprender una representación latente del audio.
- Modelos: Variational Autoencoder, Adversarial Autoencoder, Generative Adversarial Network
- Acciones: Entrenamiento, visualización del espacio latente, generación de muestras sintéticas, comparación entre las muestras sintéticas y reales mediante reducción de dimensionalidad.
- 📌 **Checkpoint 2**: Poder producir espectrogramas sintéticos usando modelos generativos.

---

### 🔹 Etapa 4 — Data augmentation generativa
- Objetivo: Mejorar el clasificador incorporando datos generados.
- Combinación de datos reales + sintéticos para entrenamiento.
- Comparación con el baseline original.
- 📌 **Checkpoint 4**: Medir la performance de los modelos entrenados con datos reales y sintéticos sobre audios reales.

---

## 📚 Resumen

- **Dataset:** Audios de ballenas vs ruido, convertidos en espectrogramas.
- **Preprocesamiento:** Normalización, resize.
- **Modelos generativos:** AE, VAE, GAN entrenados sobre espectrogramas.
- **Clasificadores:** MLP, Random Forest, Gradient Boosting, CNN.
- **Métricas:** Accuracy, F1, matriz de confusión, AUC-ROC, curvas de aprendizaj.
- **Visualización:** PCA, T-sne, curvas de aprendizaje, espectrogramas generados vs reales.


---


Este proyecto fue desarrollado como parte del curso de Aprendizaje Automático y Aprendizaje Profundo por **Martín Bianchi** y **Federico Gutman**.

