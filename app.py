import onnxruntime as rt
import pandas as pd
import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback
from google.cloud import storage

# --- 1. Configuración ---
app = Flask(__name__)
# Permitimos CORS desde cualquier origen (el "*" es el arreglo "nuclear")
CORS(app, origins="*")

# --- 2. Configuración de Google Cloud ---
# ¡CORREGIDO! Este es el nombre de tu nuevo bucket
BUCKET_NAME = "modelos-alojados" 
storage_client = storage.Client()

# Caché para no recargar los modelos en cada llamada
modelos_cargados = {} 

# --- 3. Funciones de Carga (El "corazón" de la API) ---

def get_model(model_name):
    """Descarga y carga un modelo ONNX desde GCS"""
    model_name = model_name.lower()
    if model_name in modelos_cargados:
        return modelos_cargados[model_name]
    
    model_path = f"{model_name}.onnx"
    temp_local_path = f"/tmp/{model_path}" # Ruta temporal en el servidor

    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(model_path)
        if not blob.exists():
            print(f"No se encontró {model_path} en el bucket")
            return None
        
        blob.download_to_filename(temp_local_path)
        
        # Carga el modelo con ONNX Runtime (rápido y seguro)
        model_sess = rt.InferenceSession(temp_local_path, providers=['CPUExecutionProvider'])
        
        modelos_cargados[model_name] = model_sess # Guardar en caché
        os.remove(temp_local_path) # Limpiar
        return model_sess
        
    except Exception as e:
        print(f"Error al cargar el modelo ONNX '{model_name}': {e}")
        traceback.print_exc()
        return None

def get_metadata(model_name):
    """Descarga y lee un archivo de metadatos .meta desde GCS"""
    model_name = model_name.lower()
    meta_path = f"{model_name}.meta"
    temp_local_path = f"/tmp/{meta_path}"
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(meta_path)
        if not blob.exists(): 
            print(f"No se encontró {meta_path} en el bucket")
            return None
        
        blob.download_to_filename(temp_local_path)
        with open(temp_local_path, 'r') as f:
            metadata = json.load(f)
            
        os.remove(temp_local_path)
        return metadata
    except Exception as e:
        print(f"Error al cargar metadata '{model_name}': {e}")
        return None

# --- 4. Endpoints de la API ---

@app.route('/upload', methods=['POST'])
def upload_model():
    """Endpoint para subir un modelo (.onnx) y sus metadatos (.meta)"""
    if 'model_file' not in request.files or 'meta_file' not in request.files:
         return jsonify({'error': 'Faltan "model_file" (.onnx) o "meta_file" (.meta)'}), 400
    if 'model_name' not in request.form:
         return jsonify({'error': 'Falta "model_name"'}), 400
            
    model_file = request.files['model_file']
    meta_file = request.files['meta_file']
    model_name = request.form['model_name'].lower()
    
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # Subir el .onnx
        blob_model = bucket.blob(f"{model_name}.onnx")
        blob_model.upload_from_file(model_file)
        
        # Subir el .meta
        blob_meta = bucket.blob(f"{model_name}.meta")
        blob_meta.upload_from_file(meta_file)
        
        # Limpiar caché si el modelo se actualiza
        if model_name in modelos_cargados:
            del modelos_cargados[model_name]
            
        return jsonify({'success': f"Modelo ONNX '{model_name}' y sus metadatos subidos."}), 201
    except Exception as e:
        return jsonify({'error': f"Error al guardar archivos en GCS: {e}"}), 500

@app.route('/predict/<string:model_name>', methods=['POST'])
def predict(model_name):
    """Endpoint para ejecutar una predicción"""
    model_sess = get_model(model_name.lower())
    if model_sess is None:
        return jsonify({'error': f"Modelo ONNX '{model_name}' no encontrado o no se pudo cargar."}), 404

    metadata = get_metadata(model_name.lower())
    if metadata is None:
        return jsonify({'error': f"Metadatos para '{model_name}' no encontrados (necesarios para predecir)."}), 404

    try:
        data = request.json
        features_list = data['features']
        
        # Usar los metadatos para los nombres
        column_names = [f['name'] for f in metadata['input_features']]
        
        if len(features_list) != len(column_names):
            return jsonify({'error': f"El modelo espera {len(column_names)} features, pero se recibieron {len(features_list)}."}), 400

        # Crear el DataFrame de Pandas
        df = pd.DataFrame([features_list], columns=column_names)

        # Preparar inputs para ONNX (¡Lección Aprendida!)
        onnx_input = {}
        input_info = model_sess.get_inputs()
        
        for i, input_node in enumerate(input_info):
            col_name = input_node.name
            col_data = df[col_name].values
            
            # Convertir tipos de Python a tipos de Numpy que ONNX entiende
            if input_node.type == 'tensor(string)':
                onnx_input[col_name] = col_data.reshape(-1, 1).astype(object)
            elif input_node.type == 'tensor(float)':
                onnx_input[col_name] = col_data.astype(np.float32).reshape(-1, 1)
            elif input_node.type == 'tensor(double)':
                onnx_input[col_name] = col_data.astype(np.float64).reshape(-1, 1)
            elif input_node.type == 'tensor(int64)':
                onnx_input[col_name] = col_data.astype(np.int64).reshape(-1, 1)
            else:
                 # Tipo por defecto (float32)
                onnx_input[col_name] = col_data.astype(np.float32).reshape(-1, 1)
        
        # Ejecutar la sesión de ONNX
        output_name = model_sess.get_outputs()[0].name
        prediction_onnx = model_sess.run([output_name], onnx_input)
        prediction = prediction_onnx[0] 
        
        return jsonify({
            'model_used': model_name,
            'prediction': prediction.tolist()
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': f'Error en la predicción ONNX: {str(e)}'}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """Endpoint para listar todos los modelos disponibles"""
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = bucket.list_blobs(prefix='') 
        model_names = [blob.name.replace(".meta", "") for blob in blobs if blob.name.endswith(".meta")]
        return jsonify(model_names), 200
    except Exception as e:
        return jsonify({'error': f'Error listando modelos: {e}'}), 500

@app.route('/models/<string:model_name>', methods=['GET'])
def get_model_metadata(model_name):
    """Endpoint para obtener los metadatos de un modelo"""
    metadata = get_metadata(model_name.lower())
    if metadata:
        return jsonify(metadata), 200
    else:
        return jsonify({'error': 'Metadatos no encontrados'}), 404

@app.route('/', methods=['GET'])
def root():
    return jsonify({"message": "API de Modelos v3.0 (ONNX) está activa"}), 200