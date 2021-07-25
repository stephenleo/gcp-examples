# Specifies base image and tag
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# Installs additional packages
RUN pip install tensorflow==2.4.2

# Copies the serving code to the docker image.
COPY ./app /app