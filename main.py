"""Point d'entrée : lance le pipeline d'entraînement complet.

Prérequis : un serveur MLflow doit tourner (voir README).
    mlflow server --backend-store-uri sqlite:///mlflow.db \\
                  --default-artifact-root ./mlartifacts \\
                  --host 127.0.0.1 --port 5000

Usage :
    python main.py
"""

from frauddetection.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":
    TrainingPipeline().run()
