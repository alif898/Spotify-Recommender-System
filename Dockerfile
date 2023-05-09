FROM python:3.9
LABEL authors="alif898"
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY .env .
COPY app.py .
COPY model.pkl .
COPY spotify_utilities.py .

EXPOSE 5000
CMD ["python", "app.py"]