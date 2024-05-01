FROM python:3.10
LABEL authors="Nicolas Riccieri Gardin Assumpção"

COPY requirements.txt requirements.txt

RUN pip install pip -U &&\
    pip install -r requirements.txt

COPY fpp_code ./fpp_code

WORKDIR /fpp_code/Clients

ENTRYPOINT ["python", "__init__.py"]