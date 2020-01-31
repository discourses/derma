# Dockerfile

# A python runtime base image
FROM python:3.7.5-buster

# pip
RUN pip install --upgrade pip

# If the steps of a `Dockerfile` use files that are different from the `context` file, COPY the
# file of each step separately; and RUN the file immediately after COPY
WORKDIR /app
RUN mkdir /app/checkpoints
COPY requirements.txt /app
RUN pip install --requirement /app/requirements.txt

# Specific COPY commands
COPY config.py /app/config.py
COPY src /app/src

# Port
EXPOSE 8050

# Create mountpoint
VOLUME /app/checkpoints
VOLUME /app/images

# ENTRYPOINT
ENTRYPOINT ["python"]

# CMD
CMD ["src/main.py"]