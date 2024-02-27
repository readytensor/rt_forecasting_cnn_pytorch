# CNN Forecasting Model in PyTorch

CNN Forecasting Model in PyTorch as per Ready Tensor specifications.

## Project Description

This repository is a dockerized implementation of the re-usable forecaster model. It is implemented in flexible way so that it can be used with any forecasting dataset with the use of CSV-formatted data, and a JSON-formatted data schema file. The main purpose of this repository is to provide a complete example of a machine learning model implementation that is ready for deployment.
The following are the requirements for using your data with this model:

- The data must be in CSV format.
- The schema file must contain an idField and timeField columns.
- The timeField data type must be either "DATE", "DATETIME" or "INT". [Click here for more information on dataset specifications](https://docs.readytensor.ai/category/forecasting)
- The data type of the target field, past covariates and future covariates must be NUMERIC.
- The train and test (or prediction) files must contain an ID field. The train data must also contain a target series.
- The data need to be preprocessed because the implementation assumes the data is cleaned and has no missing values.

---

### Covariates Support:

- Past ✅
- Future ❌
- Static ✅

> > This implementation treats static covariates as past covariates.

---

Here are the highlights of this implementation: <br/>

- A **CNN Forecaster** algorithm built using **PyTorch**
  Additionally, the implementation contains the following features:
- **Data Validation**: Pydantic data validation is used for the schema, training and test files, as well as the inference request data.
- **Error handling and logging**: Python's logging module is used for logging and key functions include exception handling.

## Project Structure

The following is the directory structure of the project:

- **`examples/`**: This directory contains example files for the titanic dataset. Three files are included: `smoke_test_forecasting_schema.json`, `smoke_test_forecasting_train.csv` and `smoke_test_forecasting_test.csv`. You can place these files in the `inputs/schema`, `inputs/data/training` and `inputs/data/testing` folders, respectively.
- **`model_inputs_outputs/`**: This directory contains files that are either inputs to, or outputs from, the model. When running the model locally (i.e. without using docker), this directory is used for model inputs and outputs. This directory is further divided into:
  - **`/inputs/`**: This directory contains all the input files for this project, including the `data` and `schema` files. The `data` is further divided into `testing` and `training` subsets.
  - **`/model/artifacts/`**: This directory is used to store the model artifacts, such as trained models and their parameters.
  - **`/outputs/`**: The outputs directory contains sub-directories for error logs and prediction results.
- **`src/`**: This directory holds the source code for the project. It is further divided into various subdirectories:
  - **`config/`**: for configuration files for data preprocessing, model hyperparameters, paths, etc.
  - **`data_models/`**: for data models for input validation including the schema, training and test files. It also contains the data model for the batch prediction results.
  - **`schema/`**: for schema handler script. This script contains the class that provides helper getters/methods for the data schema.
  - **`prediction/`**: Scripts for the forecaster model implemented using **Darts** library.
  - **`logger.py`**: This script contains the logger configuration using **logging** module.
  - **`train.py`**: This script is used to train the model. It loads the data, preprocesses it, trains the model, and saves the artifacts in the path `./model_inputs_outputs/model/artifacts/`.
  - **`predict.py`**: This script is used to run batch predictions using the trained model. It loads the artifacts and creates and saves the predictions in a file called `predictions.csv` in the path `./model_inputs_outputs/outputs/predictions/`.
  - **`utils.py`**: This script contains utility functions used by the other scripts.
- **`.gitignore`**: This file specifies the files and folders that should be ignored by Git.
- **`Dockerfile`**: This file is used to build the Docker image for the application.
- **`entry_point.sh`**: This file is used as the entry point for the Docker container. It is used to run the application. When the container is run using one of the commands `train`, `predict` or `serve`, this script runs the corresponding script in the `src` folder to execute the task.
- **`LICENSE`**: This file contains the license for the project.
- **`requirements.txt`** for the main code in the `src` directory
- **`README.md`**: This file (this particular document) contains the documentation for the project, explaining how to set it up and use it.

## Usage

In this section we cover the following:

- How to prepare your data for training
- How to run the model implementation locally (without Docker)
- How to run the model implementation with Docker

### Preparing your data

- If you plan to run this model implementation on your own forecasting dataset, you will need your training and testing data in a CSV format. Also, you will need to create a schema file as per the Ready Tensor specifications. The schema is in JSON format, and it's easy to create. You can use the example schema file provided in the `examples` directory as a template.

### To run locally (without Docker)

- Create your virtual environment and install dependencies listed in `requirements.txt` which is inside the `root` directory.
- Move the three example files (`smoke_test_forecasting_schema.json`, `smoke_test_forecasting_train.csv` and `smoke_test_forecasting_test.csv`) in the `examples` directory into the `./model_inputs_outputs/inputs/schema`, `./model_inputs_outputs/inputs/data/training` and `./model_inputs_outputs/inputs/data/testing` folders, respectively (or alternatively, place your custom dataset files in the same locations).
- Run the script `src/train.py` to train the forecaster model. This will save the model artifacts, including the preprocessing pipeline and label encoder, in the path `./model_inputs_outputs/model/artifacts/`.
- Run the script `src/predict.py` to run batch predictions using the trained model. This script will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `./model_inputs_outputs/outputs/predictions/`.

### To run with Docker

1. Set up a bind mount on host machine: It needs to mirror the structure of the `model_inputs_outputs` directory. Place the train data file in the `model_inputs_outputs/inputs/data/training` directory, the test data file in the `model_inputs_outputs/inputs/data/testing` directory, and the schema file in the `model_inputs_outputs/inputs/schema` directory.
2. Build the image. You can use the following command: <br/>
   `docker build -t forecaster_img .` <br/>
   Here `forecaster_img` is the name given to the container (you can choose any name).
3. Note the following before running the container for train, batch prediction:
   - The train, batch predictions tasks require a bind mount to be mounted to the path `/opt/model_inputs_outputs/` inside the container. You can use the `-v` flag to specify the bind mount.
   - When you run the train or batch prediction tasks, the container will exit by itself after the task is complete.
   - When you run training task on the container, the container will save the trained model artifacts in the specified path in the bind mount. This persists the artifacts even after the container is stopped or killed.
   - When you run the batch prediction task, the container will load the trained model artifacts from the same location in the bind mount. If the artifacts are not present, the container will exit with an error.
   - Container runs as user 1000. Provide appropriate read-write permissions to user 1000 for the bind mount. Please follow the principle of least privilege when setting permissions. The following permissions are required:
     - Read access to the `inputs` directory in the bind mount. Write or execute access is not required.
     - Read-write access to the `outputs` directory and `model` directories. Execute access is not required.
4. Run training:
   - To run training, run the container with the following command container: <br/>
     `docker run -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs forecaster_img train` <br/>
     where `forecaster_img` is the name of the container. This will train the model and save the artifacts in the `model_inputs_outputs/model/artifacts` directory in the bind mount.
5. To run batch predictions, place the prediction data file in the `model_inputs_outputs/inputs/data/testing` directory in the bind mount. Then issue the command: <br/>
   `docker run -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs forecaster_img predict` <br/>
   This will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `model_inputs_outputs/outputs/predictions/` in the bind mount.

## Requirements

Dependencies for the main model implementation in `src` are listed in the file `requirements.txt`.
You can install these packages by running the following command from the root of your project directory:

```python
pip install -r requirements.txt
```

## LICENSE

This project is provided under the MIT License. Please see the [LICENSE](LICENSE) file for more information.

## Contact Information

Repository created by Ready Tensor, Inc. (https://www.readytensor.ai/)
