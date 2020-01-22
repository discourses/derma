# Dockerfile

# A python runtime base image
FROM python:3.7.5-buster

# pip
RUN pip install --upgrade pip

# If the steps of a `Dockerfile` use files that are different from the `context` file, COPY the
# file of each step separately; and RUN the file immediately after COPY
WORKDIR /app
RUN mkdir /app/images
COPY importing/requirements.txt /app/importing
RUN pip install --requirement /app/importing/requirements.txt

# Specific COPY
COPY importing /app/importing

# Port
EXPOSE 8050

# Create mountpoint
VOLUME /app/images

# ENTRYPOINT
ENTRYPOINT ["python"]

# CMD
CMD ["importing/main.py"]