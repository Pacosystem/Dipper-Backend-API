import onnxruntime as rt
import pandas as pd
import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback
from google.cloud import storage
import threading
import time
import urllib.request

app = Flask(__name__)
CORS(app, origins="*")

BUCKET_NAME = "modelos-alojados"
storage_client = storage.Client()
modelos_cargados = {} 

# --- FUNCIONES AUXILIARES ---

def get_model(model_name):
    model_name = model_name.lower()
    if model_name in modelos_cargados:
        return modelos_cargados[model_name]
    
    model_path = f"{model_name}.onnx"
    temp_local_path = f"/tmp/{model_path}"
    
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(model_path)
        if not blob.exists(): return None
        blob.download_to_filename(temp_local_path)
        model_sess = rt.InferenceSession(temp_local_path, providers=['CPUExecutionProvider'])
        modelos_cargados[model_name] = model_sess
        os.remove(temp_local_path)
        return model_sess
    except Exception:
        return None

def get_metadata(model_name):
    model_name = model_name.lower()
    meta_path = f"{model_name}.meta"
    temp_local_path = f"/tmp/{meta_path}"
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(meta_path)
        if not blob.exists(): return None
        blob.download_to_filename(temp_local_path)
        with open(temp_local_path, 'r') as f:
            metadata = json.load(f)
        os.remove(temp_local_path)
        return metadata
    except Exception:
        return None

# --- ENDPOINT PRINCIPAL ---

@app.route('/predict/<string:model_name>', methods=['POST'])
def predict(model_name):
    """
    Endpoint con NORMALIZACIÓN AUTOMÁTICA.
    """
    model_sess = get_model(model_name.lower())
    if model_sess is None: return jsonify({'error': f"Modelo '{model_name}' no encontrado."}), 404

    metadata = get_metadata(model_name.lower())
    if metadata is None: return jsonify({'error': f"Metadatos para '{model_name}' no encontrados."}), 404

    try:
        data = request.json
        if 'features' not in data: return jsonify({'error': 'Falta "features".'}), 400
        
        features_list = data['features']
        meta_col_names = [f['name'] for f in metadata['input_features']]
        
        if len(features_list) != len(meta_col_names):
            return jsonify({'error': f"Discrepancia: Esperado {len(meta_col_names)}, Recibido {len(features_list)}."}), 400

        # Crear DataFrame
        df = pd.DataFrame([features_list], columns=meta_col_names)
        
        # ==========================================
        # 🟢 BLOQUE DE NORMALIZACIÓN (NUEVO)
        # ==========================================
        # Si el .meta tiene datos de escala, normalizamos ANTES de entrar al modelo
        if 'normalization' in metadata:
            norm = metadata['normalization']
            try:
                # 1. Normalizar Inputs (X)
                # Fórmula: (Valor - Min) / (Max - Min)
                x_min = np.array(norm['x_min'])
                x_range = np.array(norm['x_range']) # range = max - min
                
                # Aplicamos al DataFrame (asegurando orden correcto)
                x_vals = df.values.astype(np.float32)
                x_normalized = (x_vals - x_min) / x_range
                
                # Actualizamos el DataFrame con los valores 0-1
                df = pd.DataFrame(x_normalized, columns=meta_col_names)
                
            except Exception as e:
                return jsonify({'error': f"Error al normalizar inputs: {str(e)}"}), 500

        # ==========================================
        # PREPARACIÓN ONNX (Smart Mapping)
        # ==========================================
        onnx_input_nodes = model_sess.get_inputs()
        onnx_input = {}
        
        # Caso A: Modelo Vectorizado (1 entrada)
        if len(onnx_input_nodes) == 1 and len(meta_col_names) > 1:
            input_node_name = onnx_input_nodes[0].name
            combined_data = df.values.astype(np.float32)
            onnx_input[input_node_name] = combined_data
            
        # Caso B: Modelo Mapeado (N entradas)
        elif len(onnx_input_nodes) == len(meta_col_names):
            for i, node in enumerate(onnx_input_nodes):
                meta_name = meta_col_names[i]
                col_data = df[meta_name].values
                if node.type == 'tensor(string)':
                    onnx_input[node.name] = col_data.astype(object).reshape(-1, 1)
                else:
                    onnx_input[node.name] = col_data.astype(np.float32).reshape(-1, 1)
        else:
             return jsonify({'error': "Estructura incompatible Metadata vs ONNX."}), 500

        # Ejecutar Predicción
        output_name = model_sess.get_outputs()[0].name
        prediction_onnx = model_sess.run([output_name], onnx_input)
        raw_prediction = prediction_onnx[0] # Esto sale entre 0 y 1 si se usó normalización

        # ==========================================
        # 🟢 DES-NORMALIZACIÓN SALIDA (NUEVO)
        # ==========================================
        final_prediction = raw_prediction
        
        if 'normalization' in metadata:
            try:
                # 2. Des-normalizar Outputs (Y)
                # Fórmula: (Valor * Rango) + Min
                y_min = np.array(norm['y_min'])
                y_range = np.array(norm['y_range'])
                
                final_prediction = (raw_prediction * y_range) + y_min
                
            except Exception as e:
                return jsonify({'error': f"Error al des-normalizar outputs: {str(e)}"}), 500

        return jsonify({
            'model_used': model_name.lower(),
            'prediction': final_prediction.tolist(),
            'status': 'success'
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

# --- ENDPOINTS RESTANTES (Upload, List, etc.) ---
# (Mantienen la misma lógica que antes, solo asegúrate de copiar el resto del archivo anterior si lo necesitas)
@app.route('/upload', methods=['POST'])
def upload_model():
    # ... (Copia el código del upload anterior)
    if 'model_file' not in request.files or 'model_name' not in request.form: return jsonify({'error': 'Faltan datos'}), 400
    model_file = request.files['model_file']; model_name = request.form['model_name'].lower(); metadata = {}
    try:
        if 'meta_file' in request.files: metadata = json.load(request.files['meta_file'])
        elif 'input_features_json' in request.form: metadata = {"description": request.form.get('description',''), "input_features": json.loads(request.form['input_features_json'])}
        else: return jsonify({'error': 'Faltan metadatos'}), 400
        bucket = storage_client.bucket(BUCKET_NAME)
        bucket.blob(f"{model_name}.onnx").upload_from_file(model_file)
        bucket.blob(f"{model_name}.meta").upload_from_string(json.dumps(metadata, indent=2), content_type='application/json')
        if model_name in modelos_cargados: del modelos_cargados[model_name]
        return jsonify({'success': True}), 201
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    try:
        blobs = storage_client.bucket(BUCKET_NAME).list_blobs()
        return jsonify([b.name.replace('.meta','') for b in blobs if b.name.endswith('.meta')]), 200
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/models/<string:model_name>', methods=['GET'])
def get_model_details(model_name):
    meta = get_metadata(model_name)
    return jsonify(meta) if meta else (jsonify({'error': 'No encontrado'}), 404)

@app.route('/', methods=['GET'])
def root(): return jsonify({"status": "API Online v4 (Normalization)"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)