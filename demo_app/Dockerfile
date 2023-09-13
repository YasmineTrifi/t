FROM python:3-slim-buster

RUN mkdir /code

WORKDIR /code

COPY requirements.txt .

RUN python -m pip install nltk
RUN python -m nltk.downloader stopwords
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet
RUN pip install -r requirements.txt

COPY . .


CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=80"]