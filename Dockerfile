FROM tensorflow/tensorflow:2.9.1
WORKDIR /root/code/vvdiaz1/automatic_tagging/automatic_tagging/
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY .env .env
COPY setup.py setup.py
RUN pip install -e .
RUN python -c 'from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv());'
RUN python -c 'from automatic_tagging.funciones.gc_loader import load_models; models = load_models();'
EXPOSE 8000
CMD MODEL_TARGET=local uvicorn taxifare.api.fast:app --host 0.0.0.0 --port $PORT
