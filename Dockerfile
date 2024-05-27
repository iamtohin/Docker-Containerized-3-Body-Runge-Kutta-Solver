FROM python:latest
RUN pip install websockets
WORKDIR /usr/src/app
ADD solver-client.py .
CMD [ "python3", "solver-client.py" ]
