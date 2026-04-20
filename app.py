import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# ==============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================
st.set_page_config(
    page_title="AI vs Human Text Analytics",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================
# VARIABLES GLOBALES Y CACHE
# ==============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "AuthentiText_X_2026_AI_vs_Human_Detection_1K.xlsx")
TARGET = 'author_type'
NUMERIC_COLUMNS = [
    'prompt_complexity_score',
    'perplexity_score',
    'burstiness_index',
    'syntactic_variability',
    'semantic_coherence_score',
    'lexical_diversity_ratio',
    'readability_grade_level',
    'generation_confidence_score'
]

@st.cache_data
def load_data():
    """Función para procesos de Extracción (E) y Transformación (T)"""
    try:
        # Extracción
        df = pd.read_excel(DATA_PATH)
        
        # Transformación Básica
        df = df.dropna(subset=[TARGET, 'content_text'])
        df = df.drop_duplicates()
        
        # Feature Engineering
        df['text_length'] = df['content_text'].astype(str).str.len()
        
        return df
    except Exception as e:
        df_mock = pd.DataFrame({
            "author_type": ["AI", "Human", "AI", "Human", "AI"],
            "perplexity_score": [20.5, 80.2, 15.3, 75.1, 10.9],
            "burstiness_index": [0.2, 0.8, 0.1, 0.9, 0.3],
            "text_length": [1500, 2000, 800, 1800, 950]
        })
        st.warning(f"No se pudo cargar la base real. Usando datos dummy por ahora. Error: {e}")
        return df_mock

# ==============================================================
# FUNCIONES MODULARES POR SECCIÓN
# ==============================================================

def render_home(df):
    st.title("🤖 AI vs 👤 Human Text Analytics")
    st.markdown("""
    Bienvenido al **Dashboard Analítico** para la detección de texto generado por Inteligencia Artificial y humanos. 
    Este tablero es parte de un pipeline estructurado de MLOps y Data Engineering.
    """)
    
    st.subheader("Métricas Clave del Dataset")
    col1, col2, col3, col4 = st.columns(4)
    
    total_docs = len(df)
    ai_docs = len(df[df[TARGET] == 'AI']) if TARGET in df.columns else 0
    human_docs = len(df[df[TARGET] == 'Human']) if TARGET in df.columns else 0
    avg_len = df['text_length'].mean() if 'text_length' in df.columns else 0
    
    with col1:
        st.metric("Total Documentos", f"{total_docs:,}")
    with col2:
        st.metric("Generados por IA", f"{ai_docs:,}")
    with col3:
        st.metric("Escritos por Humanos", f"{human_docs:,}")
    with col4:
        st.metric("Longitud Promedio (caracteres)", f"{avg_len:,.0f}")
        
    st.info("Navega mediante la barra lateral (Sidebar) para explorar las diferentes etapas del análisis estadístico y predictivo.")

def render_exploration(df):
    st.title("Exploración de Datos Interactiva 🔍")
    
    st.markdown("### Filtros Dinámicos")
    col1, col2 = st.columns(2)
    
    with col1:
        sel_author = st.multiselect("Filtrar por Tipo de Autor", options=df[TARGET].unique(), default=df[TARGET].unique())
    with col2:
        if 'readability_grade_level' in df.columns:
            min_grade, max_grade = int(df['readability_grade_level'].min()), int(df['readability_grade_level'].max())
            grade_range = st.slider("Filtro: Nivel de Legibilidad", min_grade, max_grade, (min_grade, max_grade))
        else:
            grade_range = None
            
    # Aplicar filtros
    filtered_df = df[df[TARGET].isin(sel_author)]
    if grade_range and 'readability_grade_level' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['readability_grade_level'] >= grade_range[0]) & 
            (filtered_df['readability_grade_level'] <= grade_range[1])
        ]
        
    st.markdown("### Vista Previa de Datos")
    st.dataframe(filtered_df.head(100), use_container_width=True)
    
    st.markdown("### Tabla de Frecuencias (Tipo Autor)")
    freq_table = filtered_df[TARGET].value_counts().reset_index()
    freq_table.columns = ['Autor', 'Frecuencia']
    freq_table['Porcentaje (%)'] = round((freq_table['Frecuencia'] / len(filtered_df)) * 100, 2)
    st.table(freq_table)

def render_visualizations(df):
    st.title("Visualización de Métricas 📊")
    
    col_sel = st.selectbox("Selecciona métrica para analizar su distribución", [c for c in NUMERIC_COLUMNS if c in df.columns] + ['text_length'])
    
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(
            df, x=col_sel, color=TARGET, 
            opacity=0.7, marginal="box",
            title=f"Distribución de {col_sel} por Autor",
            barmode="overlay"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col2:
        fig_box = px.box(
            df, x=TARGET, y=col_sel, color=TARGET,
            title=f"Boxplot de {col_sel}",
            points="all"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Matriz de Correlación")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto", 
        color_continuous_scale="RdBu_r",
        title="Correlación entre métricas lingüísticas"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

def render_prediction(df):
    st.title("Análisis Predictivo en Tiempo Real 🤖")
    st.markdown("Basado en el dataset actual, entrenaremos un modelo `RandomForestClassifier` para clasificar y permitiremos hacer una predicción interactiva.")
    
    # Preparación simple
    features = [c for c in NUMERIC_COLUMNS if c in df.columns]
    
    if not features:
        st.warning("No hay suficientes features para entrenar.")
        return
        
    X = df[features]
    y = df[TARGET]
    
    # Label encoding básico asumiendo 'AI' y 'Human'
    y = y.map({"AI": 0, "Human": 1})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    st.success(f"Modelo Entrenado Automáticamente ✅ | Accuracy en Test: **{acc*100:.2f}%**")
    
    st.markdown("### Simulador de Clasificación")
    st.markdown("Ajusta los valores para simular si un texto es IA o Humano")
    
    user_inputs = {}
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            min_val = float(df[feature].min())
            max_val = float(df[feature].max())
            mean_val = float(df[feature].mean())
            user_inputs[feature] = st.number_input(f"{feature}", min_value=min_val, max_value=max_val, value=mean_val)
            
    if st.button("Analizar Texto Simulado", type="primary"):
        input_data = pd.DataFrame([user_inputs])
        prediction = clf.predict(input_data)[0]
        prob = clf.predict_proba(input_data)[0]
        
        result_text = "HUMANO 👤" if prediction == 1 else "INTELIGENCIA ARTIFICIAL 🤖"
        st.subheader(f"El modelo clasifica el texto como: **{result_text}**")
        
        fig_prob = go.Figure(data=[go.Pie(labels=['IA', 'Humano'], values=[prob[0], prob[1]], hole=.4)])
        fig_prob.update_layout(title_text="Confianza del Modelo")
        st.plotly_chart(fig_prob)

def render_airflow():
    st.title("Orquestación de Datos: DAGs (Simulación Airflow) 🌬️")
    
    st.markdown("""
    En un entorno real de Ingeniería de Datos corporativo, los flujos (ETL) no se corren manualmente desde un script, sino que se orquestan. 
    A continuación se documentan 3 ejercicios de **Directed Acyclic Graphs (DAGs)** que componen este flujo.
    """)
    
    st.info("Nota técnica: Estos DAGs son ilustrativos y representan la lógica separada del pipeline en herramientas como Apache Airflow.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### DAG 1: Ingesta Cruda")
        st.markdown("""
        - **Extracción horaria:** `fetch_kaggle_data`
        - **Objetivo:** Conectarse a la API, descargar el .xlsx crudo.
        - **Almacenamiento:** Lo guarda temporalmente en un bucket s3 (Ej: `s3://data-raw/ai_human`).
        """)
        st.code("ext_task >> val_raw_task", language="python")
        
    with col2:
        st.markdown("#### DAG 2: Transformación (ETL)")
        st.markdown("""
        - **Trigger:** Al concluir el DAG 1.
        - **Limpieza:** Dropear nulos (`clean_nulls_task`).
        - **Transformación:** Calcular `text_length` y escalar variables numéricas.
        - **Alerta:** Enviar a Slack si se detectan anomolías (Quality check).
        """)
        st.code("read_s3 >> clean >> feat_eng >> write_s3_clean", language="python")

    with col3:
        st.markdown("#### DAG 3: Reentrenamiento Modelo")
        st.markdown("""
        - **Frecuencia:** Semanal.
        - **Ejecución:** Descarga la data procesada.
        - **Model ops:** Separa en Train/Test, entrena Random Forest y guarda métricas (MLflow).
        - **Deploy:** Actualiza el registro (Model Registry).
        """)
        st.code("fetch_clean >> train_model >> register_model", language="python")

# ==============================================================
# ROUTER PRINCIPAL
# ==============================================================
def main():
    df = load_data()
    
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1620712943543-bcc4688e7485?q=80&w=300&auto=format&fit=crop", caption="Data Pipeline")
        st.title("Navegación")
        seccion = st.radio(
            "Ir a la sección:",
            ("Inicio", "Exploración de datos", "Visualizaciones", "Análisis predictivo", "Orquestación (Airflow)")
        )
        
        st.markdown("---")
        st.markdown("👨‍💻 Desarrollado como proyecto Senior de Datos.")
        
    if seccion == "Inicio":
        render_home(df)
    elif seccion == "Exploración de datos":
        render_exploration(df)
    elif seccion == "Visualizaciones":
        render_visualizations(df)
    elif seccion == "Análisis predictivo":
        render_prediction(df)
    elif seccion == "Orquestación (Airflow)":
        render_airflow()

if __name__ == "__main__":
    main()
