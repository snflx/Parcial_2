# AI Generated vs Human Written Text - Proyecto Analítica ETL

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=for-the-badge&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas)
![Airflow](https://img.shields.io/badge/Airflow-ETL_Simulado-017CEE?style=for-the-badge&logo=apache-airflow)

Bienvenido a este proyecto de ingeniería de datos y visualización analítica. Este repositorio implementa un pipeline simulado y una aplicación interactiva en Streamlit para el análisis profundo del dataset **"AI Generated vs Human Written Text Dataset"**.

## Objetivo del Proyecto

El objetivo es proveer una plataforma interactiva, estructurada de manera modular y escalable (evitando código spaghetti) para diferenciar características lingüísticas y métricas entre el texto escrito por humanos frente al generado por modelos de IA (gpt-4, gemini, claude, etc). Las métricas incluyen perplejidad, índice de 'burstiness', coherencia semántica, entre otras.

---

## 🏗️ Arquitectura del Proyecto

```text
├── app/
│   ├── app.py                 # Aplicación principal de Streamlit
│   └── requirements.txt       # Dependencias del proyecto
├── data/
│   └── AuthentiText_...xlsx   # Dataset origen
├── docs/
│   └── index.html             # Landing page optimizada para GitHub Pages (Documentación)
├── notebooks/
│   └── EDA_Analysis.ipynb     # Notebook Jupyter con Exploratory Data Analysis
└── README.md                  # Documentación técnica (este archivo)
```

---

## 🔄 Pipeline ETL Simulado (Integración tipo Airflow)

Aunque localmente el código carga un CSV/Excel directamente, el diseño de la lógica obedece al esquema generalizado de un DAG en Apache Airflow. El flujo lógico consiste en:

### 1. Ingesta (Extract)
- **Origen:** Obtención periódica del dataset o carga manual del archivo Excel (`AuthentiText_X_2026_AI_vs_Human_Detection_1K.xlsx`).
- **Validación:** Comprobación rápida de existencia de columnas esenciales y tipos del esquema de datos.

### 2. Transformación (Transform)
- **Limpieza:** Dropear registros duplicados.
- **Manejo de nulos:** Rellenar o descartar valores sin target o sin 'content_text'.
- **Normalización y Feature Engineering:** Generación de nueva variable `text_length` calculada a partir de los caracteres del texto de la columna `content_text`. Conversión de formatos y tipado correcto.

### 3. Carga (Load)
- **Almacenamiento:** Guardar el DataFrame estructurado y en memoria listo para alimentar el pipeline del Dashboard (Streamlit) o para el entrenamiento de un modelo predictivo (Scikit-Learn).

---

## 👨‍💻 Buenas Prácticas Implementadas

- **Diseño por Capas:** Separación entre interfaz (`app.py`), análisis (`notebooks`), y recursos (`data/`).
- **Componentización:** Uso de funciones discretas para cada módulo (ETL, Visualización, Predicción).
- **Control de errores:** Uso de bloques Try/Except en funciones críticas, como la carga y transformación.
- **Visualización dinámica:** Plotly y Seaborn integrados para analítica moderna, evitando gráficos estáticos y sin opciones de interpretación por el usuario final.


