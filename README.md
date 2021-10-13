# azureml-demo

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
