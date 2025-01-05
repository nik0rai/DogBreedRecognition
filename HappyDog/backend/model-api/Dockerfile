FROM python:3.8-slim as base

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

ENV FLASK_APP=app.py

#region Add development with debug

FROM base as dev

RUN pip install debugpy

EXPOSE 5000 5678

CMD python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m flask run -h 0.0.0.0 -p 5000

#endregion Add development with debug

#region Add production

FROM base as prod

EXPOSE 5000 

CMD flask run -h 0.0.0.0 -p 5000

#endregion Add production