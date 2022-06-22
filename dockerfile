FROM python:3.8

WORKDIR /PviewServer

COPY req.txt .

RUN pip install -r req.txt

RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx

COPY . .

EXPOSE 5000

CMD ["python3", "main.py"]