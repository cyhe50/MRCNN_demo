FROM waleedka/modern-deep-learning

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .
ENV PYTHONPATH = /usr/local/lib/python3.5/dist-packages
CMD ["python3","demo.py"]