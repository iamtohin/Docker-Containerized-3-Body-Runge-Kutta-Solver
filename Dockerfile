FROM python:latest
RUN pip install websockets pandas
WORKDIR /server/
ADD solver-api.py .
EXPOSE 8001
CMD [ "python3", "-u", "solver-api.py"]
