"""Étape 2 — Transformation : construit le préprocesseur (mise à l'échelle).

On garde volontairement le préprocesseur SÉPARÉ ici pour illustrer le concept,
mais à l'étape suivante on le glisse dans une Pipeline scikit-learn avec le
modèle. Avantage : le modèle enregistré dans MLflow est auto-suffisant — il
prend des features brutes en entrée, la mise à l'échelle est embarquée.
"""

import sys

from sklearn.preprocessing import StandardScaler

from frauddetection.exception.exception import FraudDetectionException
from frauddetection.logging.logger import logger


class DataTransformation:
    @staticmethod
    def build_preprocessor() -> StandardScaler:
        try:
            logger.info("Construction du préprocesseur : StandardScaler")
            return StandardScaler()
        except Exception as e:
            raise FraudDetectionException(e, sys) from e
