# Specifies base image and tag
FROM tensorflow/tensorflow:2.4.2-gpu
WORKDIR /root

# Installs additional packages
RUN pip install tqdm

# Copies the trainer code to the docker image.
COPY trainer/model.py /root/model.py
COPY trainer/task.py /root/task.py
COPY trainer/utils.py /root/utils.py

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "task.py"]