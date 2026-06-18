# Détection de fraude bancaire — démo cycle de vie MLflow

Projet end-to-end illustrant comment **MLflow** structure le cycle de vie d'un
modèle de ML, de l'expérimentation au modèle promu en production. Cas fil rouge :
la détection de fraude sur des transactions bancaires.

## L'histoire

Un modèle est « en production » : un `DecisionTree(max_depth=2)`. Il affiche
**98,8 % d'accuracy** — donc tout va bien, non ? En réalité son **recall est de
0,00** : il ne détecte **aucune** fraude. Sur un jeu où la fraude ne représente
que ~1,2 % des transactions, prédire « légitime » partout suffit à avoir une
accuracy flatteuse.

On entraîne un `RandomForest(class_weight="balanced")` : **recall 0,90**. Sur la
même transaction suspecte (4 200 € depuis Varsovie, 2 h du matin, appareil
inconnu), le verdict bascule :

| Modèle | Accuracy | Recall | F1 | Score sur la fraude évidente | Verdict |
|---|---|---|---|---|---|
| v1 — DecisionTree(max_depth=2) | 0,988 | **0,000** | 0,000 | 0,365 | ❌ légitime |
| v2 — RandomForest(balanced) | 0,998 | **0,903** | 0,926 | 0,677 | 🚨 fraude |

MLflow rend ce piège visible (tracking), versionne les modèles (registry) et
permet de promouvoir le bon modèle **sans toucher au code applicatif** (alias).

## Structure

```
fraud-detection-mlops/
├── frauddetection/            # package principal
│   ├── components/            # data_ingestion, data_transformation, model_trainer
│   ├── constants/             # noms de features, modèle, alias, chemins
│   ├── entity/                # config_entity (entrées), artifact_entity (sorties)
│   ├── pipeline/              # training_pipeline (orchestration)
│   ├── utils/                 # calcul des métriques
│   ├── exception/ · logging/  # gestion d'erreurs et logs
├── scripts/
│   ├── generate_data.py       # génère le dataset synthétique
│   └── promote_model.py       # déplace l'alias champion (promotion live)
├── notebooks/
│   └── demo_mlflow.ipynb      # déroulé pas à pas de la démo
├── data_schema/schema.yaml
├── fraud_data/transactions.csv
├── app.py                     # application Streamlit (charge @champion)
├── main.py                    # lance le pipeline d'entraînement
├── Dockerfile · requirements.txt · setup.py
```

## Lancer la démo en local

> Python 3.10+. Installer les dépendances : `pip install -r requirements.txt`

**1. Démarrer le serveur MLflow** (tracking + registry, backend SQLite) :

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 --port 5000
```

L'UI est alors disponible sur http://127.0.0.1:5000.

**2. (Optionnel) Régénérer les données** — un CSV est déjà fourni :

```bash
python scripts/generate_data.py
```

**3. Entraîner les deux modèles** (les logge, les enregistre, pose `champion` sur la v1) :

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000   # Windows : set MLFLOW_TRACKING_URI=...
python main.py
```

**4. Lancer l'application** (dans un autre terminal, même variable d'env) :

```bash
streamlit run app.py
```

Teste les deux scénarios : la fraude évidente passe pour **légitime** (v1 en prod).

**5. Promouvoir le bon modèle — le moment clé** :

```bash
python scripts/promote_model.py 2
```

Rafraîchis l'app : le badge passe à `v2 · RandomForestClassifier` et la même
transaction est désormais détectée comme **fraude**. On peut aussi déplacer
l'alias directement dans l'UI MLflow (onglet *Models* → *Aliases*).

## Note technique : alias, pas « stages »

Les *stages* `Staging`/`Production` du Model Registry sont **dépréciés depuis
MLflow 2.9** et voués à disparaître. Ce projet utilise l'approche moderne : des
**alias** de version (`@champion`) et des **tags**. L'application cible donc :

```
models:/FraudDetector@champion
```

Promouvoir un modèle = déplacer cet alias. Rien d'autre ne change.

## Déploiement sur EC2

L'app et le serveur MLflow peuvent cohabiter sur une même instance EC2 :

- ouvrir les ports **8501** (Streamlit) et **5000** (MLflow) dans le *security group* ;
- lancer le serveur MLflow puis `streamlit run app.py --server.address=0.0.0.0` ;
- accéder via `http://<IP-publique-EC2>:8501`.

Un `Dockerfile` est fourni. Passer l'URL du tracking via la variable d'env :

```bash
docker build -t fraud-app .
docker run -e MLFLOW_TRACKING_URI=http://<host>:5000 -p 8501:8501 fraud-app
```
