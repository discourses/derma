# Dockerfile

# A python runtime base image
FROM python:3.7.5-buster

# pip
RUN pip install --upgrade pip

# If the steps of a `Dockerfile` use files that are different from the `context` file, COPY the
# file of each step separately; and RUN the file immediately after COPY
WORKDIR /app
COPY requirements.txt /app
RUN pip install --requirement /app/requirements.txt

# Specific COPY
COPY config.py /app/config.py
COPY src/data /app/src
COPY src/evaluation /app/src
COPY src/modelling/extraction app/src/

# Port
EXPOSE 8050

# Create mountpoint
VOLUME /app/images

# ENTRYPOINT
ENTRYPOINT ["python"]

# CMD
CMD ["src/main.py"]