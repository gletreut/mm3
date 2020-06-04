# Running MM3 with docker

You can run mm3 scripts using the docker by building the image using the supplied [Dockerfile](docker/Dockerfile). An easy way to build the image is to resort to the [docker-compose.yml](docker-compose.yml) file:
```
$ docker-compose build
```

This command will create an image `mm3:latest`. MM3 commands can now be run as:
```
$ docker run --rm -it -u $(id -u):$(id -g) -w=/user_data -v $PWD:/user_data -v /path/to/mm3:/mm3 -h mm3 mm3:latest /mm3/mm3_script.py [OPTIONS] ARGUMENTS
```
just as you would run:
```
$ python3 mm3_script.py [OPTIONS] ARGUMENTS
```

You might want to create an alias. For example `alias docker-mm3=\'docker run --rm -it -u $(id -u):$(id -g) -w=/user_data -v $PWD:/user_data -v /path/to/mm3:/mm3 -h mm3\'`.

## ND2 file processing
1. docker-mm3 /mm3/mm3_nd2ToTiFF.py -f params_processing.yaml
2. docker-mm3 /mm3/mm3_Compile.py -f params_processing.yaml
3. Channel picking -- non-interactive
  1. Execute the command "docker-mm3 /mm3/mm3_ChannelPicker.py -f params_processing.yaml -i"
  2. Copy and edit the file `analysis/specs.yaml`. Use the pdf files exported in `analysis/fovs`.
  3. Re-execute the channel picking command, supplying the edited `specs.yaml` file and loading the previously computed correlations. `docker-mm3 /mm3/mm3_ChannelPicker.py -f params_processing.yaml -i -c -s specs.yaml`.
4. Subtraction. `docker-mm3 /mm3/mm3_Subtract.py -f params_processing.yaml`.
