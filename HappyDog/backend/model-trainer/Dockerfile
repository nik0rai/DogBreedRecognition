FROM python:3.8-slim as base

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

#region Add development with debug

FROM base as dev

RUN pip install debugpy

EXPOSE 6000 6678

CMD python -m debugpy --listen 0.0.0.0:6000 --wait-for-client app.py

#endregion Add development with debug

#region Add production

FROM base as prod

EXPOSE 6000 

CMD flask run -h 0.0.0.0 -p 6000

#endregion Add production