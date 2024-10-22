import requests
import json
import os
from datetime import datetime
import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from joblib import dump

# OpenWeatherMap API key
API_KEY = 'ad58e0536b2e863043d708d6e2b6e7fd' #depuis la site OpenWeatherMap
CITIES = ['paris', 'london', 'washington']

# Task 1: collecter la données weather et la stocker dans /app/raw_files
def fetch_weather_data(**kwargs):
    raw_data_dir = '/app/raw_files'
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)

    weather_data = []
    for city in CITIES:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            weather_data.append(response.json())
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    file_name = f'{raw_data_dir}/{timestamp}.json'
    
    with open(file_name, 'w') as file:
        json.dump(weather_data, file)

# Task 2: Transformer les derniers  20 files en CSV
def transform_data_to_csv(n_files=20, filename='data.csv'):
    parent_folder = '/app/raw_files'
    files = sorted(os.listdir(parent_folder), reverse=True)[:n_files]
    dfs = []
    
    for f in files:
        with open(os.path.join(parent_folder, f), 'r') as file:
            data_temp = json.load(file)
        for data_city in data_temp:
            dfs.append({
                'temperature': data_city['main']['temp'],
                'city': data_city['name'],
                'pressure': data_city['main']['pressure'],
                'date': f.split('.')[0]
            })
    
    df = pd.DataFrame(dfs)
    df.to_csv(f'/app/clean_data/{filename}', index=False)

# Task 3: Transformer toutes la données en CSV
def transform_all_data_to_fulldata_csv(filename='fulldata.csv'):
    parent_folder = '/app/raw_files'
    files = sorted(os.listdir(parent_folder), reverse=True)
    dfs = []
    
    for f in files:
        with open(os.path.join(parent_folder, f), 'r') as file:
            data_temp = json.load(file)
        for data_city in data_temp:
            dfs.append({
                'temperature': data_city['main']['temp'],
                'city': data_city['name'],
                'pressure': data_city['main']['pressure'],
                'date': f.split('.')[0]
            })
    
    df = pd.DataFrame(dfs)
    df.to_csv(f'/app/clean_data/{filename}', index=False)

# Task 4: Entrainement desmodeles etcalcul des scores
def compute_model_score(model, X, y):
    cross_validation = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
    return cross_validation.mean()

def train_models(**kwargs):
    # Prepare data
    df = pd.read_csv('/app/clean_data/fulldata.csv')
    df = df.sort_values(['city', 'date'], ascending=True)
    
    # Create features and target
    df['target'] = df['temperature'].shift(1)
    df = df.dropna()
    X = df.drop(['target', 'date'], axis=1)
    y = df['target']

    # Train models and store their scores
    score_lr = compute_model_score(LinearRegression(), X, y)
    score_dt = compute_model_score(DecisionTreeRegressor(), X, y)
    score_rf = compute_model_score(RandomForestRegressor(), X, y)

    # Push the scores via XCom
    kwargs['ti'].xcom_push(key='model_scores', value={
        'LinearRegression': score_lr,
        'DecisionTree': score_dt,
        'RandomForest': score_rf
    })

# Task 5: Selectionner the best model, rentrainer et lenregistrer

def select_and_save_best_model(**kwargs):
    # Pull the scores from XCom
    scores = kwargs['ti'].xcom_pull(key='model_scores', task_ids='train_models')
    
    # Select the best model
    best_model_name = min(scores, key=scores.get)
    df = pd.read_csv('/app/clean_data/fulldata.csv')
    df = df.sort_values(['city', 'date'], ascending=True)
    
    df['target'] = df['temperature'].shift(1)
    df = df.dropna()
    X = df.drop(['target', 'date'], axis=1)
    y = df['target']

    if best_model_name == 'LinearRegression':
        model = LinearRegression()
    elif best_model_name == 'DecisionTree':
        model = DecisionTreeRegressor()
    else:
        model = RandomForestRegressor()
    
    model.fit(X, y)
    model_path = f'/app/models/{best_model_name}.pkl'
    dump(model, model_path)
    print(f'Model {best_model_name} saved to {model_path}')

# Définition du DAG
default_args = {
    'owner': 'user',
    'start_date': days_ago(1),
    'retries': 1
}

with DAG(
    dag_id='weather_data_pipeline',
    default_args=default_args,
    description='A data pipeline to process weather data and update the dashboard',
    schedule_interval='* * * * *',  # Toute les minutes
    catchup=False
) as dag:

    # Fetch weather data(Task1)
    fetch_weather = PythonOperator(
        task_id='fetch_weather_data',
        python_callable=fetch_weather_data,
        provide_context=True
    )

    # Transform the last 20 files into data.csv (Task2)
    transform_to_csv = PythonOperator(
        task_id='transform_to_csv',
        python_callable=transform_data_to_csv
    )

    # Transform all data into fulldata.csv (Task3)
    transform_to_fulldata_csv = PythonOperator(
        task_id='transform_to_fulldata_csv',
        python_callable=transform_all_data_to_fulldata_csv
    )

   # Task to train models (Task4)
    train_models = PythonOperator(
        task_id='train_models',
        python_callable=train_models,
        provide_context=True
    )

    # Task to select and save the best model(Task5)
    select_and_save_best_model = PythonOperator(
        task_id='select_and_save_best_model',
        python_callable=select_and_save_best_model,
        provide_context=True
    )


    # Task dependencies
    fetch_weather >> transform_to_csv >> transform_to_fulldata_csv >> train_models >> select_and_save_best_model
