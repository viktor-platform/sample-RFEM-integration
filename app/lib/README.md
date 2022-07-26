# Worker

The files in this folder are used by the [worker](https://docs.viktor.ai/docs/guides/integrations/#worker). For this application to work, you need to [install](https://docs.viktor.ai/docs/worker/) a generic worker anywhere on your computer. The worker will also interact with the RFEM webserver. You will need to keep RFEM running and enable the web services in the program settings of RFEM. Move the files in `app\lib` over to the same location as the viktor-worker-generic.exe (default location for worker is C:\Program Files\Viktor\Viktor worker for generic v5.1.0).

## create_model.bat

The `create_model.bat` is the script that the worker is going to run. The script finishes if an output.json is created. Be sure that python is installed so this script can call it. If the command `python` does not work change line 2 of `create_model.bat` to the right command (e.g. `python3`, `python3.10`).

## config.yaml

The config file tells the worker what to execute. To configure this file you need to change the path to where `create_model.bat` is located. `workingDirectoryPath` should be the parent directory of `create_model.bat`.

## build_rfem_model.py

This file will build the model and parses the calculations to the worker. You don't need to do anything with this file.
