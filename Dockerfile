FROM python:3.8.18-bookworm

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

COPY ./requirements.txt /app/requirements.txt
COPY ./setup.py /app/setup.py
RUN pip install -r /app/requirements.txt

RUN apt-get update && apt install python3-tk -y

CMD ["bash"]