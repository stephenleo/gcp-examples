# Specifies base image and tag
FROM tensorflow/tensorflow:2.4.2
WORKDIR /root

# Installs additional packages
RUN pip install Flask

# Copies the trainer code to the docker image.
COPY app.py /root/app.py

# Set the environment variable for flask
ENV FLASK_APP=app.py
EXPOSE 5000

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["flask", "run", "--host=0.0.0.0"]