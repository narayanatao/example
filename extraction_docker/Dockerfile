# FROM ubuntu:latest

FROM python:3.7-slim-stretch

RUN apt-get update -y
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
ENV DEBIAN_FRONTEND=nonintercative
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install --no-cache-dir --upgrade pip

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt
# RUN python3 -m spacy download en_core_web_sm
RUN pip install python-Levenshtein

#Expose ports
ENV LISTEN_PORT=8094
EXPOSE 8094

# copy the content of the local src directory to the working directory
COPY src/ .


# command to run on container start
CMD [ "python3", "./__init__.py" ]