1- Primeiramente o python necessita do uso de um ambiente virtual (venv) para isolar as dependências do projeto. Para criar e ativar o venv, no terminal, execute os seguintes comandos:

Para criar venv no Ubuntu:

python3 -m venv ./venv

Para ativar o venv:

source ./venv/bin/activate

Para desativar o venv:

deactivate

Para criar venv no windows:

python3 -m venv venv  

Para ativar o venv:

venv\Scripts\activate

2- Instale as dependências:

pip install -r requirements.txt

3- Execução da api:

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Abra seu navegador e acesse: http://localhost:8000/docs