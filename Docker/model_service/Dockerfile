FROM python:3.9

#
WORKDIR /code

#
RUN pip install --upgrade pip
RUN pip install poetry

#
COPY ./Docker/model_service/pyproject.toml /code/

COPY ../../src/app /code/app
COPY ./Docker/model_service/.env /code/app/.env


RUN poetry config virtualenvs.create false \
  && poetry install $(test "$YOUR_ENV" == production && echo "--no-dev") --no-interaction --no-ansi
  
  
CMD ["uvicorn", "app.inference:app", "--host", "0.0.0.0", "--port", "80"]


