FROM python:3.8.12-buster
WORKDIR /root/code/vvdiaz1/automatic_tagging/automatic_tagging/
COPY automatic_tagging automatic_tagging
COPY requirements.txt requirements.txt
COPY .env .env
COPY setup.py setup.py
RUN pip install --upgrade pip
RUN pip install .
RUN python -c 'from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv());\
                from automatic_tagging.funciones.more_loaders import model_loaders; model_loaders();'
EXPOSE 8000
CMD uvicorn automatic_tagging.api.fast:app --host 0.0.0.0 --port 8000
