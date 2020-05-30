# Running MM3 with docker

You can run mm3 scripts using the docker by building the image using the supplied [Dockerfile](docker/Dockerfile). An easy way to build the image is to resort to the [docker-compose.yml](docker-compose.yml) file:
```
$ docker-compose build
```

This command will create an image `mm3:latest`. MM3 command can now be run as:
```
$ docker run --rm -it -u $(id -u):$(id -g) -w=/user_data -v $PWD:/user_data -v /path/to/mm3:/mm3 -h mm3 mm3:latest /mm3/mm3_script.py [OPTIONS] ARGUMENTS
```
just as you would run:
```
$ python3 mm3_script.py [OPTIONS] ARGUMENTS
```

You might want to create an alias for `docker run --rm -it -u $(id -u):$(id -g) -w=/user_data -v $PWD:/user_data -v /path/to/mm3:/mm3 -h mm3`.
