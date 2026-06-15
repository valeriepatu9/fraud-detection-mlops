FROM python:3.11-slim

WORKDIR /app

# Dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code + données
COPY . .

# Streamlit écoute sur 8501
EXPOSE 8501

# L'app a besoin de joindre le serveur MLflow. Sur EC2, soit MLflow tourne
# dans le même conteneur/host, soit on passe l'URL via la variable d'env :
#   docker run -e MLFLOW_TRACKING_URI=http://<host>:5000 -p 8501:8501 fraud-app
ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5000

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
