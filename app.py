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
CORS(app, origins="*")
BUCKET_NAME = "modelos-alojados" # ¡Tu bucket!
storage_client = storage.Client()
modelos_cargados = {} 

# --- 2. Funciones de Carga (Sin cambios) ---
# (get_model y get_metadata siguen igual)
def get_model(model_name):
    model_name = model_name.lower()
    if model_name in modelos_cargados:
        return modelos_cargados[model_name]
    model_path = f"{model_name}.onnx"
    temp_local_path = f"/tmp/{model_path}"
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(model_path)
        if not blob.exists():
            print(f"No se encontró {model_path} en el bucket")
            return None
        blob.download_to_filename(temp_local_path)
        model_sess = rt.InferenceSession(temp_local_path, providers=['CPUExecutionProvider'])
        modelos_cargados[model_name] = model_sess
        os.remove(temp_local_path)
        return model_sess
    except Exception as e:
        print(f"Error al cargar el modelo ONNX '{model_name}': {e}")
        return None

def get_metadata(model_name):
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

# --- 3. Endpoints de la API ---

# --- ¡¡AQUÍ ESTÁ EL CAMBIO IMPORTANTE!! ---
@app.route('/upload', methods=['POST'])
def upload_model():
    """
    Endpoint flexible: acepta metadatos como un archivo .meta 
    O como campos de formulario (JSON string).
    """
    # 1. Validar campos comunes
    if 'model_file' not in request.files:
         return jsonify({'error': 'Falta "model_file" (.onnx)'}), 400
    if 'model_name' not in request.form:
         return jsonify({'error': 'Falta "model_name"'}), 400
            
    model_file = request.files['model_file']
    model_name = request.form['model_name'].lower()
    metadata = {}
    
    try:
        # 2. Revisa qué método de metadatos se usó
        if 'meta_file' in request.files:
            # --- Opción A: Se subió un archivo .meta ---
            meta_file = request.files['meta_file']
            # Cargar el JSON del archivo
            metadata = json.load(meta_file)
            
        elif 'input_features_json' in request.form:
            # --- Opción B: Se usó el formulario manual ---
            metadata = {
                "description": request.form.get('description', 'Sin descripción'),
                "input_features": json.loads(request.form['input_features_json'])
            }
        else:
            # Si no se envió ninguno de los dos
            return jsonify({'error': 'Faltan los metadatos (ni "meta_file" ni "input_features_json")'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Error procesando metadatos: {e}'}), 400

    try:
        # 3. Guardar los archivos en GCS
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # Subir el .onnx
        blob_model = bucket.blob(f"{model_name}.onnx")
        blob_model.upload_from_file(model_file)
        
        # Subir el .meta (construido desde Opción A o B)
        blob_meta = bucket.blob(f"{model_name}.meta")
        blob_meta.upload_from_string(json.dumps(metadata, indent=2), content_type='application/json')
        
        if model_name in modelos_cargados:
            del modelos_cargados[model_name]
            
        return jsonify({'success': f"Modelo ONNX '{model_name}' y sus metadatos subidos."}), 201
    
    except Exception as e:
        return jsonify({'error': f"Error al guardar archivos en GCS: {e}"}), 500

# --- (El resto de los endpoints /predict, /models, etc. no cambian) ---

@app.route('/predict/<string:model_name>', methods=['POST'])
def predict(model_name):
    # (Esta función no cambia)
    model_sess = get_model(model_name.lower())
    if model_sess is None: return jsonify({'error': f"Modelo ONNX '{model_name}' no encontrado."}), 404
    metadata = get_metadata(model_name.lower())
    if metadata is None: return jsonify({'error': f"Metadatos para '{model_name}' no encontrados."}), 404
    try:
        data = request.json; features_list = data['features']
        column_names = [f['name'] for f in metadata['input_features']]
        if len(features_list) != len(column_names): return jsonify({'error': f"Discrepancia de features."}), 400
        df = pd.DataFrame([features_list], columns=column_names)
        onnx_input = {}
        input_info = model_sess.get_inputs()
        for i, input_node in enumerate(input_info):
            col_name = input_node.name; col_data = df[col_name].values
            if input_node.type == 'tensor(string)': onnx_input[col_name] = col_data.reshape(-1, 1).astype(object)
            else: onnx_input[col_name] = col_data.astype(np.float32).reshape(-1, 1)
        output_name = model_sess.get_outputs()[0].name
        prediction_onnx = model_sess.run([output_name], onnx_input)
        prediction = prediction_onnx[0] 
        return jsonify({'model_used': model_name, 'prediction': prediction.tolist()})
    except Exception as e:
        print(traceback.format_exc()); return jsonify({'error': f'Error en la predicción: {str(e)}'}), 500

@app.route('/models', methods=['GET'])
def list_models():
    # (Esta función no cambia)
    try:
        bucket = storage_client.bucket(BUCKET_NAME); blobs = bucket.list_blobs(prefix='') 
        model_names = [blob.name.replace(".meta", "") for blob in blobs if blob.name.endswith(".meta")]
        return jsonify(model_names), 200
    except Exception as e: return jsonify({'error': f'Error listando modelos: {e}'}), 500

@app.route('/models/<string:model_name>', methods=['GET'])
def get_model_metadata(model_name):
    # (Esta función no cambia)
    metadata = get_metadata(model_name.lower());
    if metadata: return jsonify(metadata), 200
    else: return jsonify({'error': 'Metadatos no encontrados'}), 404

@app.route('/', methods=['GET'])
def root():
    return jsonify({"message": "API de Modelos v3.2 (Híbrida) está activa"}), 200