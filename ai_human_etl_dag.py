import os
from datetime import datetime, timedelta
# Simulación de importaciones de Airflow (en entorno real estarían instaladas)
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.email import EmailOperator
    from airflow.operators.empty import EmptyOperator
except ImportError:
    # Dummy classes para evitar errores al leer el archivo sin airflow instalado localmente
    class DAG:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args, **kwargs): pass
    class PythonOperator:
        def __init__(self, *args, **kwargs): pass
    class EmailOperator:
        def __init__(self, *args, **kwargs): pass
    class EmptyOperator:
        def __init__(self, *args, **kwargs): pass

# ==============================================================
# RESPUESTAS Y ADAPTACIÓN DE LAS PREGUNTAS DEL EXAMEN A ESTE DAG
# ==============================================================
# Pregunta 1 adaptada (Timeouts de API): 
# Q: Si la tarea extract_data falla por timeout de la API, ¿qué comportamiento se espera según la configuración de default_args?
# R: Esperamos que Airflow aplique la regla de 'retries' definida en `default_args`. 
#    Por ejemplo, abajo está configurado en 3 reintentos antes de fallar definitivamente.

# Pregunta 2 adaptada (Nuevas tareas de carga reales):
# Q: Si un estudiante quiere agregar una nueva tarea que cargue los datos en BigQuery en lugar de solo simular la carga, ¿qué cambios debería hacer?
# R: Debería reemplazar la función de python_callable por el uso del `BigQueryInsertJobOperator` o un PythonOperator 
#    con la librería `google-cloud-bigquery`, pasando las credenciales correspondientes e insertando la data transformada.

# Pregunta 3 adaptada (Alertas y Trigger Rules):
# Q: En el DAG de monitoreo, la tarea alert_email se configura con trigger_rule="one_failed". ¿Qué significa esto?
# R: Significa que la tarea se ejecutará inmediatamente si *al menos una* de las tareas predecesoras (extracción o transformación)
#    falla, enviando así una notificación de error al equipo de Data Engineers sin esperar que termine todo el DAG.

# ==============================================================

default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,                           # RESPUESTA PREGUNTA 1: Intentará la descarga 3 veces si da timeout
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ai_human_etl_dag',
    default_args=default_args,
    description='DAG ETL completo para el anÃ¡lisis de AI vs Human',
    schedule_interval=timedelta(days=1),    # Programación diaria
    start_date=datetime(2026, 4, 1),
    catchup=False,
    tags=['text_analytics', 'nlp'],
) as dag:

    # 1. Tarea de Ingesta (Extracción)
    def extract_data(**kwargs):
        """Simula la descarga de los datos fuente. Podría dar timeout."""
        print("Extrayendo AuthentiText_X_2026_AI_vs_Human_Detection_1K.xlsx desde el bucket de origen...")
        # Lógica real de requests/api

    task_extract = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
        execution_timeout=timedelta(minutes=2) # Timeout simulado
    )

    # 2. Tarea de Transformación (Quality Checks y Métricas)
    def transform_data(**kwargs):
        """Calcula el text_length y evita errores de división por cero."""
        print("Transformando datos, limpiando nulos y duplicados...")

    task_transform = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data,
    )

    # 3. Tarea de Carga (Simulada para uso local)
    def load_data(**kwargs):
        """Simula cargar la base transformada en un DataWarehouse"""
        print("Guardando datos limpios en la capa Silver/Gold...")

    task_load = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
    )

    # 4. Tarea de Integración a BigQuery (Respuesta a la pregunta del estudiante)
    # En la práctica se sustituiría task_load por algo parecido a:
    task_load_bigquery = EmptyOperator(
        task_id='load_to_bigquery_real',
        doc_md="Conector y lógica real a BigQuery (Ej. GCSToBigQueryOperator)"
    )

    # 5. Monitoreo: Alerta en caso de falla con trigger_rule especial
    task_alert_email = EmailOperator(
        task_id='alert_email',
        to='data_team@empresa.com',
        subject='[ALERTA] Fallo crítico en el pipeline de Text Analytics',
        html_content='Se detactó una falla en el proceso de Ingesta o Transformación.',
        trigger_rule='one_failed' # RESPUESTA PREGUNTA 3: Ejecuta si extraer o transformar falla
    )

    # Definición del flujo:
    # Si la extracción triunfa, se pasa a la transformación, y luego a la carga.
    # Si cualquiera de ellas falla, se activa la alerta.
    task_extract >> task_transform >> [task_load, task_load_bigquery]
    [task_extract, task_transform, task_load] >> task_alert_email
