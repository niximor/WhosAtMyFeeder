FROM python:3.8
WORKDIR /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt .
RUN pip install --default-timeout 100 -r requirements.txt
COPY model.tflite .
COPY birdnames.db .
COPY speciesid.py .
COPY webui.py .
COPY queries.py .
COPY templates/ ./templates/
COPY static/ ./static/

CMD python ./speciesid.py
