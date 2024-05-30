FROM python:latest 

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["streamlit", "run", "classifier_app.py"]