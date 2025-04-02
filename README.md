# ggis598
Graduate Capstone Project

## Setup

```shell
pyenv virtualenv 3.12.9 ggis598
pyenv activate ggis598
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -r galileo/requirements-dev.txt
pip install -e galileo
export PYTHONPATH="$PYTHONPATH:$(pwd)/galileo"
```

To verify setup worked, the following should print a useful documentation message:

```shell
python -c "import single_file_galileo; print(help(single_file_galileo))"
```
