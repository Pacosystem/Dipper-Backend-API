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

# ==========================================
# 1. CONFIGURACIÓN INICIAL
# ==========================================
app = Flask(__name__)
# Permitir CORS para que tu Front-end pueda llamar a esta API sin bloqueos
CORS(app, origins="*")

# Nombre del bucket en Google Cloud Storage donde se guardan los modelos
BUCKET_NAME = "modelos-alojados"
storage_client = storage.Client()

# Caché en memoria para almacenar sesiones de modelos cargados y no re-descargarlos
# Formato: { 'nombre_modelo': onnxruntime.InferenceSession }
modelos_cargados = {} 

# ==========================================
# 2. FUNCIONES AUXILIARES (Gestión de GCS)
# ==========================================

def get_model(model_name):
    """
    Busca el modelo en caché. Si no está, lo descarga de GCS, 
    lo carga en memoria y lo guarda en caché.
    """
    model_name = model_name.lower()
    
    # 1. Verificar Caché (Memoria RAM)
    if model_name in modelos_cargados:
        return modelos_cargados[model_name]
    
    # 2. Si no está, preparar descarga
    model_path = f"{model_name}.onnx"
    temp_local_path = f"/tmp/{model_path}"
    
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(model_path)
        
        if not blob.exists():
            print(f"Error: No se encontró el archivo {model_path} en el bucket.")
            return None
            
        # Descargar a directorio temporal del sistema
        blob.download_to_filename(temp_local_path)
        
        # 3. Crear sesión de inferencia ONNX
        # 'providers' define si usar CPU o GPU (CUDA). Usamos CPU por compatibilidad general.
        model_sess = rt.InferenceSession(temp_local_path, providers=['CPUExecutionProvider'])
        
        # Guardar en caché global
        modelos_cargados[model_name] = model_sess
        
        # Limpiar archivo temporal
        os.remove(temp_local_path)
        
        return model_sess
        
    except Exception as e:
        print(f"Error crítico al cargar el modelo ONNX '{model_name}': {e}")
        # Intentar limpiar si quedó algo a medias
        if os.path.exists(temp_local_path):
            os.remove(temp_local_path)
        return None

def get_metadata(model_name):
    """
    Descarga y lee el archivo .meta asociado al modelo desde GCS.
    Este archivo contiene los nombres de las columnas, tipos y descripciones.
    """
    model_name = model_name.lower()
    meta_path = f"{model_name}.meta"
    temp_local_path = f"/tmp/{meta_path}"
    
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(meta_path)
        
        if not blob.exists(): 
            print(f"Advertencia: No se encontró {meta_path} en el bucket.")
            return None
            
        blob.download_to_filename(temp_local_path)
        
        with open(temp_local_path, 'r') as f:
            metadata = json.load(f)
            
        os.remove(temp_local_path)
        return metadata
        
    except Exception as e:
        print(f"Error al cargar metadata para '{model_name}': {e}")
        return None

# ==========================================
# 3. ENDPOINTS DE LA API
# ==========================================

@app.route('/predict/<string:model_name>', methods=['POST'])
def predict(model_name):
    """
    Endpoint PRINCIPAL: Ejecuta predicciones.
    Lógica Inteligente: Se adapta a modelos que piden 1 vector o N variables.
    """
    
    # A. Cargar Modelo y Metadata
    model_sess = get_model(model_name.lower())
    if model_sess is None:
        return jsonify({'error': f"Modelo '{model_name}' no disponible o no encontrado."}), 404

    metadata = get_metadata(model_name.lower())
    if metadata is None:
        return jsonify({'error': f"Faltan los metadatos (.meta) para '{model_name}'."}), 404

    try:
        # B. Leer datos de la petición
        data = request.json
        if not data or 'features' not in data:
             return jsonify({'error': 'JSON inválido. Se requiere el campo "features".'}), 400
             
        features_list = data['features']
        
        # Obtener nombres de columnas esperados desde el .meta
        # (Se asume que el orden de la lista 'features' coincide con 'input_features')
        meta_col_names = [f['name'] for f in metadata['input_features']]
        
        # Validación básica de longitud
        if len(features_list) != len(meta_col_names):
            return jsonify({
                'error': f"Discrepancia de datos: El modelo espera {len(meta_col_names)} valores ({meta_col_names}), "
                         f"pero recibió {len(features_list)}."
            }), 400

        # Crear DataFrame temporal para facilitar el manejo de datos
        df = pd.DataFrame([features_list], columns=meta_col_names)

        # C. Lógica de Mapeo Inteligente (Smart Mapping)
        onnx_input_nodes = model_sess.get_inputs()
        onnx_input = {}
        
        # --- CASO 1: Modelo "Vectorizado" (Keras/TensorFlow Export) ---
        # El modelo ONNX tiene 1 sola entrada (matriz), pero el usuario envió N variables.
        # Acción: Combinar todas las variables en una sola matriz [1, N].
        if len(onnx_input_nodes) == 1 and len(meta_col_names) > 1:
            input_node_name = onnx_input_nodes[0].name
            
            # Convertimos todo a float32 y extraemos la matriz numpy
            # IMPORTANTE: Esto asume que todas las entradas son numéricas.
            try:
                combined_data = df.values.astype(np.float32)
            except ValueError:
                 return jsonify({'error': 'Error de conversión: El modelo vectorizado espera solo números.'}), 400
            
            onnx_input[input_node_name] = combined_data
            
        # --- CASO 2: Modelo "Mapeado 1 a 1" (Sklearn / Custom) ---
        # El modelo ONNX tiene tantas entradas como variables hay en metadata.
        # Acción: Asignar cada columna del DataFrame a su nodo correspondiente en ONNX.
        elif len(onnx_input_nodes) == len(meta_col_names):
            for i, node in enumerate(onnx_input_nodes):
                # Asumimos coincidencia por posición (índice 0 metadata -> índice 0 modelo)
                meta_name = meta_col_names[i]
                col_data = df[meta_name].values
                
                # Manejo de tipos de datos (Strings vs Floats)
                if node.type == 'tensor(string)':
                    onnx_input[node.name] = col_data.astype(object).reshape(-1, 1)
                else:
                    try:
                        onnx_input[node.name] = col_data.astype(np.float32).reshape(-1, 1)
                    except ValueError:
                         return jsonify({'error': f"El valor para '{meta_name}' no es un número válido."}), 400
        
        # --- CASO 3: Error Estructural ---
        else:
             return jsonify({
                 'error': f"Conflicto de estructura: Metadata define {len(meta_col_names)} variables, "
                          f"pero el modelo ONNX espera {len(onnx_input_nodes)} entradas. No se puede inferir el mapeo automáticamente."
             }), 500

        # D. Ejecutar Predicción
        output_name = model_sess.get_outputs()[0].name
        prediction_onnx = model_sess.run([output_name], onnx_input)
        
        # Aplanar resultado para que sea serializable en JSON
        prediction_result = prediction_onnx[0].tolist()

        return jsonify({
            'model_used': model_name.lower(),
            'prediction': prediction_result,
            'status': 'success'
        })

    except Exception as e:
        # Capturar el error completo para debug en consola
        print(traceback.format_exc())
        return jsonify({'error': f'Error interno del servidor al predecir: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload_model():
    """
    Sube un nuevo modelo (.onnx) y sus metadatos (.meta o JSON) al bucket.
    """
    # Validaciones básicas
    if 'model_file' not in request.files:
         return jsonify({'error': 'Falta el archivo del modelo (key: model_file)'}), 400
    if 'model_name' not in request.form:
         return jsonify({'error': 'Falta el nombre del modelo (key: model_name)'}), 400
            
    model_file = request.files['model_file']
    model_name = request.form['model_name'].lower() # Normalizar a minúsculas
    metadata = {}
    
    try:
        # A. Detectar fuente de metadatos
        if 'meta_file' in request.files:
            # Opción 1: Archivo .meta subido directamente
            metadata = json.load(request.files['meta_file'])
        elif 'input_features_json' in request.form:
            # Opción 2: JSON string enviado desde formulario web
            try:
                metadata = {
                    "description": request.form.get('description', 'Modelo subido sin descripción'),
                    "input_features": json.loads(request.form['input_features_json'])
                }
            except json.JSONDecodeError:
                return jsonify({'error': 'El campo input_features_json no es un JSON válido'}), 400
        else:
            return jsonify({'error': 'Faltan metadatos. Envíe un archivo .meta o input_features_json'}), 400
            
        # B. Subir a Google Cloud Storage
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # 1. Subir .onnx
        blob_onnx = bucket.blob(f"{model_name}.onnx")
        blob_onnx.upload_from_file(model_file)
        
        # 2. Subir .meta
        blob_meta = bucket.blob(f"{model_name}.meta")
        blob_meta.upload_from_string(
            json.dumps(metadata, indent=2), 
            content_type='application/json'
        )
        
        # C. Limpiar caché (si estamos actualizando un modelo existente)
        if model_name in modelos_cargados:
            del modelos_cargados[model_name]
            print(f"Modelo '{model_name}' eliminado de caché para recarga.")
            
        return jsonify({
            'success': True,
            'message': f"Modelo '{model_name}' subido y publicado correctamente."
        }), 201
    
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': f"Error al subir archivos: {str(e)}"}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """Lista todos los modelos disponibles (aquellos que tienen archivo .meta)."""
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blobs = bucket.list_blobs()
        
        # Filtramos solo los archivos que terminan en .meta para obtener la lista limpia
        models = [b.name.replace('.meta', '') for b in blobs if b.name.endswith('.meta')]
        
        return jsonify(models), 200
    except Exception as e:
        return jsonify({'error': f"Error listando modelos: {str(e)}"}), 500

@app.route('/models/<string:model_name>', methods=['GET'])
def get_model_details(model_name):
    """Devuelve la información detallada (JSON) de un modelo específico."""
    meta = get_metadata(model_name)
    if meta:
        return jsonify(meta), 200
    return jsonify({'error': 'Modelo no encontrado o sin metadatos'}), 404

@app.route('/', methods=['GET'])
def root():
    """Health check simple."""
    return jsonify({
        "status": "online", 
        "version": "3.5",
        "system": "API de Inferencia ONNX (Smart Mapping)"
    }), 200

if __name__ == '__main__':
    # Configuración de puerto para despliegues en Cloud Run, App Engine o Docker
    port = int(os.environ.get('PORT', 8080))
    
    # --- AUTO-VERIFICACIÓN DE SALUD (NUEVO) ---
    def verify_deployment():
        # Esperar un poco a que el servidor de Flask arranque por completo
        time.sleep(3)
        url = f"http://127.0.0.1:{port}/"
        try:
            print(f"🔍 [Self-Check] Verificando salud de la API en {url}...")
            # Usamos urllib para no depender de 'requests' si no estuviera instalado
            with urllib.request.urlopen(url) as response:
                if response.getcode() == 200:
                    data = json.loads(response.read().decode())
                    print(f"✅ [Self-Check] API RESPONDE CORRECTAMENTE: {data}")
                else:
                    print(f"⚠️ [Self-Check] API respondió con código inesperado: {response.getcode()}")
        except Exception as e:
            print(f"❌ [Self-Check] Error al contactar la API: {e}")
            print("   (Esto es normal si estás en un entorno Serverless donde localhost no es accesible)")

    # Ejecutar la verificación en un hilo separado (Daemon) para no bloquear el inicio de la app
    threading.Thread(target=verify_deployment, daemon=True).start()

    app.run(host='0.0.0.0', port=port)