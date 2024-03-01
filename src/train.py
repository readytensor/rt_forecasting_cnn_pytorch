import time

from config import paths
from data_models.data_validator import validate_data
from logger import get_logger, log_error
from prediction.predictor_model import (
    save_predictor_model,
    train_predictor_model,
)
from preprocessing.preprocess import (
    get_preprocessing_pipelines,
    fit_transform_with_pipeline,
    save_pipelines,
)
from schema.data_schema import load_json_data_schema, save_schema
from utils import (
    read_csv_in_directory,
    read_json_as_dict,
    set_seeds,
    ResourceTracker,
)

logger = get_logger(task_name="train")


def run_training(
    input_schema_dir: str = paths.INPUT_SCHEMA_DIR,
    saved_schema_dir_path: str = paths.SAVED_SCHEMA_DIR_PATH,
    model_config_file_path: str = paths.MODEL_CONFIG_FILE_PATH,
    train_dir: str = paths.TRAIN_DIR,
    preprocessing_config_file_path: str = paths.PREPROCESSING_CONFIG_FILE_PATH,
    preprocessing_dir_path: str = paths.PREPROCESSING_DIR_PATH,
    predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
    default_hyperparameters_file_path: str = paths.DEFAULT_HYPERPARAMETERS_FILE_PATH,
) -> None:
    """
    Run the training process and saves model artifacts

    Args:
        input_schema_dir (str, optional): The directory path of the input schema.
        saved_schema_dir_path (str, optional): The path where to save the schema.
        model_config_file_path (str, optional): The path of the model
            configuration file.
        train_dir (str, optional): The directory path of the train data.
        predictor_dir_path (str, optional): Dir path where to save the
            predictor model.
        default_hyperparameters_file_path (str, optional): The path of the default
            hyperparameters file.
    Returns:
        None
    """

    try:
        with ResourceTracker(logger=logger, monitoring_interval=5):
            logger.info("Starting training...")
            # load and save schema
            logger.info("Loading and saving schema...")
            data_schema = load_json_data_schema(input_schema_dir)
            save_schema(schema=data_schema, save_dir_path=saved_schema_dir_path)

            # load model config
            logger.info("Loading model config...")
            model_config = read_json_as_dict(model_config_file_path)

            # set seeds
            logger.info("Setting seeds...")
            set_seeds(seed_value=model_config["seed_value"])

            # load train data
            logger.info("Loading train data...")
            train_data = read_csv_in_directory(train_dir)

            # validate the data
            logger.info("Validating train data...")
            validated_data = validate_data(
                data=train_data, data_schema=data_schema, is_train=True
            )

            logger.info("Loading preprocessing config...")
            preprocessing_config = read_json_as_dict(preprocessing_config_file_path)

            # use default hyperparameters to train model
            logger.info("Loading hyperparameters...")
            default_hyperparameters = read_json_as_dict(
                default_hyperparameters_file_path
            )

            # fit and transform using pipeline and target encoder, then save them
            logger.info("Training preprocessing pipeline...")
            training_pipeline, inference_pipeline, encode_len = (
                get_preprocessing_pipelines(
                    data_schema,
                    validated_data,
                    preprocessing_config,
                    default_hyperparameters,
                )
            )
            trained_pipeline, transformed_data = fit_transform_with_pipeline(
                training_pipeline, validated_data
            )
            print("Transformed training data shape:", transformed_data.shape)

            # # use default hyperparameters to train model
            logger.info("Training forecaster...")
            forecaster = train_predictor_model(
                history=transformed_data,
                forecast_length=data_schema.forecast_length,
                hyperparameters=default_hyperparameters,
            )

        # Save pipelines
        logger.info("Saving pipelines...")
        save_pipelines(trained_pipeline, inference_pipeline, preprocessing_dir_path)

        # save predictor model
        logger.info("Saving forecaster...")
        save_predictor_model(forecaster, predictor_dir_path)

    except Exception as exc:
        err_msg = "Error occurred during training."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.TRAIN_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_training()
