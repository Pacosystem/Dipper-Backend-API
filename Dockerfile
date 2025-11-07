# Usar una imagen base de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar todo (app.py, requirements.txt, etc.)
COPY . .

# Instalar las librerías
RUN pip install --no-cache-dir -r requirements.txt

# Comando final para ejecutar la API (con el arreglo de $PORT)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT app:app"]
