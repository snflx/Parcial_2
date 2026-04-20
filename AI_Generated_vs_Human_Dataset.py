# =========================================
# 0. CONFIGURACIÓN
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

sns.set(style="whitegrid")

# =========================================
# 1. CARGA DE DATOS
# =========================================
df = pd.read_csv("AuthentiText_X_2026_AI_vs_Human_Detection_1K.csv")

# =========================================
# 2. VARIABLES
# =========================================
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

TARGET = 'author_type'

# =========================================
# UTILIDADES
# =========================================
def print_section(title):
    print("\n" + "="*60)
    print(title.upper())
    print("="*60)

# =========================================
# 3. INSPECCIÓN GENERAL
# =========================================
def basic_info(df):
    print_section("Información general")
    print(df.info())
    print("\nShape:", df.shape)
    print("\nDistribución de clases:")
    print(df[TARGET].value_counts())

# =========================================
# 4. ESTADÍSTICA DESCRIPTIVA
# =========================================
def descriptive_stats(df):
    print_section("Estadísticas descriptivas")
    print(df[NUMERIC_COLUMNS].describe())

# =========================================
# 5. ANÁLISIS POR CLASE (AI vs HUMAN)
# =========================================
def group_analysis(df):
    print_section("Comparación AI vs Human")
    print(df.groupby(TARGET)[NUMERIC_COLUMNS].mean())

# =========================================
# 6. ASIMETRÍA Y CURTOSIS
# =========================================
def shape_analysis(df):
    print_section("Asimetría y curtosis")

    for col in NUMERIC_COLUMNS:
        data = df[col].dropna()
        print(f"\n{col}:")
        print(f"  Skewness: {stats.skew(data):.4f}")
        print(f"  Kurtosis: {stats.kurtosis(data):.4f}")

# =========================================
# 7. VISUALIZACIONES
# =========================================
def plot_distributions(df):
    print_section("Histogramas")

    for col in NUMERIC_COLUMNS:
        plt.figure(figsize=(6,4))
        sns.histplot(data=df, x=col, hue=TARGET, kde=True)
        plt.title(f"Distribución de {col}")
        plt.show()

def boxplots(df):
    print_section("Boxplots")

    for col in NUMERIC_COLUMNS:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=TARGET, y=col, data=df)
        plt.title(f"{col} por tipo de autor")
        plt.show()

def correlation_heatmap(df):
    print_section("Correlación")

    plt.figure(figsize=(10,8))
    sns.heatmap(df[NUMERIC_COLUMNS].corr(), annot=True, cmap='coolwarm')
    plt.title("Matriz de correlación")
    plt.show()

def scatter_analysis(df):
    print_section("Relaciones clave")

    sns.scatterplot(
        x='perplexity_score',
        y='burstiness_index',
        hue=TARGET,
        data=df
    )
    plt.title("Perplexity vs Burstiness")
    plt.show()

# =========================================
# 8. FEATURE ENGINEERING
# =========================================
def feature_engineering(df):
    print_section("Feature Engineering")

    df['text_length'] = df['content_text'].str.len()

    print("Nueva variable creada: text_length")
    return df

# =========================================
# 9. EJECUCIÓN
# =========================================
def run_analysis(df):
    basic_info(df)
    descriptive_stats(df)
    group_analysis(df)
    shape_analysis(df)

    df = feature_engineering(df)

    plot_distributions(df)
    boxplots(df)
    correlation_heatmap(df)
    scatter_analysis(df)

    print_section("Análisis finalizado")

# RUN
run_analysis(df)