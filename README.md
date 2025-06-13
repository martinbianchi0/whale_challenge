# 🐋 Whale Sound Classification with Generative Models

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
- Acciones: Visualización de espectrogramas, inspección de clases, balance de datos.

---

### 🔹 Etapa 1 — Clasificador baseline
- Objetivo: Establecer un punto de partida con datos reales.
- Preprocesamiento: conversión a espectrogramas o extracción de features acústicos.
- Modelos: MLP o CNN simple.
- Métricas: accuracy, F1-score, precision, recall.
- 📌 **Checkpoint 1**: "Con datos reales y un modelo simple, logramos X% de performance."

---

### 🔹 Etapa 2 — Modelado generativo (AE / VAE / GAN)
- Objetivo: Aprender una representación latente del audio.
- Modelos: Autoencoder, Variational Autoencoder, opcional GAN.
- Acciones: Entrenamiento, visualización del espacio latente, generación de muestras sintéticas.
- 📌 **Checkpoint 2**: "Ya podemos generar datos sintéticos de calidad razonable."

---

### 🔹 Etapa 3 — Clasificación desde espacio latente
- Objetivo: Evaluar si el espacio latente tiene valor informativo.
- Clasificación directamente desde vectores latentes.
- Clustering y análisis de similaridad en el espacio latente.
- 📌 **Checkpoint 3**: "El espacio latente captura estructura útil para clasificación y agrupamiento."

---

### 🔹 Etapa 4 — Data augmentation generativa
- Objetivo: Mejorar el clasificador incorporando datos generados.
- Combinación de datos reales + sintéticos para entrenamiento.
- Comparación con el baseline original.
- 📌 **Checkpoint 4**: "La data generada mejora / no mejora la clasificación."

---

### 🔹 Etapa 5 — Robustez del modelo generativo *(opcional)*
- Objetivo: Medir estabilidad ante perturbaciones.
- Entrenamiento con ruido.
- Comparación de latentes y performance.
- 📌 **Checkpoint 5**: "El modelo generativo mantiene / pierde estabilidad frente al ruido."

---

### 🔹 Etapa 6 — Transfer Learning
- Objetivo: Reutilizar modelos entrenados para un nuevo dataset de mamíferos.
- Uso del encoder como extractor de features.
- Comparación contra entrenamiento desde cero.
- 📌 **Checkpoint 6**: "El conocimiento del modelo se transfiere con éxito / no aporta."

---

## 📚 Metodología

- **Dataset:** Audios de ballenas vs ruido, convertidos en espectrogramas o representaciones acústicas.
- **Preprocesamiento:** Normalización, resize, extracción de MFCC o STFT.
- **Modelos generativos:** AE, VAE, GAN entrenados sobre espectrogramas.
- **Clasificadores:** MLP, SVM, CNN.
- **Métricas:** Accuracy, F1, precision, recall, distancia en espacio latente.
- **Visualización:** PCA, t-SNE, curvas de aprendizaje, espectrogramas generados vs reales.

---

## 📁 Estructura del proyecto


---

## 🎨 Visualizaciones sugeridas

- Espacio latente (2D PCA o t-SNE)
- Spectrogramas reales vs generados
- Curvas de aprendizaje (loss, accuracy)
- Clusters en espacio latente
- Comparación de métricas con y sin data generada

---

## ✍️ Contribuciones

Este proyecto fue desarrollado como parte del curso de Fundamentos de Inteligencia Artificial (FIA - UdeSA, 2024) por **Martín Bianchi**.

---

## 🧪 Resultado esperado

Validar si los modelos generativos pueden **mejorar el rendimiento de clasificación** en tareas de audio, y estudiar la **utilidad del espacio latente** para tareas de agrupamiento y transferencia.

---