FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install flask pandas torch scikit-learn

EXPOSE 5000

CMD ["python", "app.py"]
