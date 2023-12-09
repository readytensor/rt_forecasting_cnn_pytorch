from typing import List

import numpy as np
import pandas as pd
import sys

from config import paths
from data_models.data_validator import validate_data
from data_models.prediction_data_model import validate_predictions
from logger import get_logger, log_error
from prediction.predictor_model import load_predictor_model, predict_with_model
from preprocessing.preprocess import (
    load_pipeline_of_type,
    fit_transform_with_pipeline,
    inverse_scale_predictions
)
from schema.data_schema import load_saved_schema
from utils import (
    read_csv_in_directory,
    read_json_as_dict,
    save_dataframe_as_csv,
    cast_time_col
)

logger = get_logger(task_name="predict")


def create_predictions_dataframe(
        pred_input: pd.DataFrame,
        predictions_arr: np.ndarray,
        prediction_field_name: str,
        id_field_name: str,
        time_field_name: str
    ) -> pd.DataFrame:
    """
    Converts the predictions numpy array into a dataframe having the required structure.

    Args:
        pred_input (pd.DataFrame): Test data input.
        predictions_arr (np.ndarray): Forecast from forecasting model.
        prediction_field_name (str): Field name to use for forecast values.
        id_field_name (str): Name for the id field.
        time_field_name (str): Name for the time field.

    Returns:
        Predictions as a pandas dataframe
    """
    N_train, T_train, _ = predictions_arr.shape
    N_test = pred_input[id_field_name].unique().shape[0]
    T_test = pred_input[time_field_name].unique().shape[0]
    if N_train != N_test:
        raise ValueError(f"Number of series in test input ({N_test}) does not match"
                         f"# of series in train data ({N_train})")
    if T_train != T_test:
        raise ValueError(f"Length of series in test input ({N_test}) does not match"
                         f"expected forecast window length ({N_train})")
    
    predictions_df = pred_input[[id_field_name, time_field_name]].copy()
    predictions_df.sort_values(by=[id_field_name, time_field_name], inplace=True)
    predictions_df[prediction_field_name] = np.squeeze(predictions_arr).flatten()
    return predictions_df


def run_batch_predictions(
    saved_schema_dir_path: str = paths.SAVED_SCHEMA_DIR_PATH,
    model_config_file_path: str = paths.MODEL_CONFIG_FILE_PATH,
    train_dir: str = paths.TRAIN_DIR,
    test_dir: str = paths.TEST_DIR,
    preprocessing_dir_path: str = paths.PREPROCESSING_DIR_PATH,
    predictor_dir_path: str = paths.PREDICTOR_DIR_PATH,
    predictions_file_path: str = paths.PREDICTIONS_FILE_PATH,
) -> None:
    """
    Run batch predictions on test data, save the predicted probabilities to a CSV file.

    This function reads test data from the specified directory,
    loads the preprocessing pipeline and pre-trained predictor model,
    transforms the test data using the pipeline,
    makes predictions using the trained predictor model,
    adds ids into the predictions dataframe,
    and saves the predictions as a CSV file.

    Args:
        saved_schema_dir_path (str): Dir path to the saved data schema.
        model_config_file_path (str): Path to the model configuration file.
        train_dir (str): Directory path for the train data.
        test_dir (str): Directory path for the test data.
        preprocessing_dir_path (str): Path to the saved pipeline file.
        predictor_file_path (str): Path to the saved predictor model file.
        predictions_file_path (str): Path where the predictions file will be saved.
    """

    try:
        logger.info("Making batch predictions...")

        logger.info("Loading schema...")
        data_schema = load_saved_schema(saved_schema_dir_path)

        logger.info("Loading model config...")
        model_config = read_json_as_dict(model_config_file_path)

        # we need history to make predictions
        logger.info("Loading training data...")
        train_data = read_csv_in_directory(file_dir_path=train_dir)
        logger.info("Validating training data...")
        validated_train_data = validate_data(
            data=train_data, data_schema=data_schema, is_train=True
        )
        
        # we need the test data to return our final predictions with right columns
        logger.info("Loading test data...")
        test_data = read_csv_in_directory(file_dir_path=test_dir)
        test_data = cast_time_col(
            test_data, data_schema.time_col, data_schema.time_col_dtype)
        logger.info("Validating test data...")
        validated_test_data = validate_data(
            data=test_data, data_schema=data_schema, is_train=False
        )

        # fit and transform using pipeline and target encoder, then save them
        logger.info("Loading preprocessing pipeline ...")
        inference_pipeline = load_pipeline_of_type(
            preprocessing_dir_path,
            pipeline_type="inference"
        )
        _, transformed_train_data = fit_transform_with_pipeline(
            inference_pipeline, validated_train_data
        )

        logger.info("Loading predictor model...")
        predictor_model = load_predictor_model(predictor_dir_path)

        logger.info("Making predictions...")
        predictions_arr = predict_with_model(
            predictor_model,
            transformed_train_data
        )

        logger.info("Rescaling predictions...")
        rescaled_preds_arr = inverse_scale_predictions(
            predictions_arr, inference_pipeline
        )

        logger.info("Creating final predictions dataframe...")
        predictions_df = create_predictions_dataframe(
            pred_input=validated_test_data,
            predictions_arr=rescaled_preds_arr,
            prediction_field_name=model_config["prediction_field_name"],
            id_field_name=data_schema.id_col,
            time_field_name=data_schema.time_col
        )

        logger.info("Validating predictions dataframe...")
        validated_predictions = validate_predictions(
            predictions_df, data_schema, model_config["prediction_field_name"]
        )

        logger.info("Saving predictions dataframe...")
        save_dataframe_as_csv(
            dataframe=validated_predictions, file_path=predictions_file_path
        )

        logger.info("Batch predictions completed successfully")

    except Exception as exc:
        err_msg = "Error occurred during prediction."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.PREDICT_ERROR_FILE_PATH)
        # re-raise the error
        raise Exception(f"{err_msg} Error: {str(exc)}") from exc


if __name__ == "__main__":
    run_batch_predictions()
