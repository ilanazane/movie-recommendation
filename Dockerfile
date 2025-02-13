# use official python image as base 
FROM python:3.9

# set working directory 
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./movie_model.pkl /code/movie_model.pkl
COPY ./data /code/data


RUN pip install --no-cache-dir -r /code/requirements.txt

# copy the current directory's files into the container 
COPY ./src /code/src

EXPOSE 8000

# define command to run 
# CMD ["uvicorn", "src.app:app", "--port", "5000"]
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]


# in command line: 
# docker build -t image_name .
# docker run --name container_name -p 8000:8000 image_name 
# maybe dont use the name and container name 