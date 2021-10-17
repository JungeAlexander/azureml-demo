# azureml-demo

🚧🚧🚧 WIP 🚧🚧🚧

## Environment setup

```shell
python -m virtualenv .venv
source .venv/bin/activate
python -m pip install -f requirements.txt
# TO ENSURE EXACT PINNED PACKAGE VERSIONS:
# python -m pip install -f requirements.lock
```

Updating pinned package versions:

```
pip freeze > requirements.lock
```

## Workflow

### Register dataset

### Train model

- Register environment
- Track performance -> MLflow?
- model dependencies? e.g. tunetuning, downstream use

### Register model

### Deploy to Azure Function

- https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-functions


### Fancier model

- keras CTC/transformers model
- wav2vec