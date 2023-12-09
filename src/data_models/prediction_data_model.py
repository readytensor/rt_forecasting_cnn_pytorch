import numpy as np
import pandas as pd
from pydantic import BaseModel, validator

from schema.data_schema import ForecastingSchema


def get_predictions_validator(
        schema: ForecastingSchema, prediction_field_name: str,
    ) -> BaseModel:
    """
    Returns a dynamic Pydantic data validator class based on the provided schema.

    The resulting validator checks the following:

    1. That the predictions dataFrame is not empty.
    2. That the predictions dataFrame contains the ID field specified in the schema.
    3. That the ID field does not contain nulls.
    4. That the predictions dataFrame contains the time field specified in the schema.
    5. That the time field does not contain nulls.
    6. That the id and time field combinations are unique.
    7. That the predictions dataFrame contains the target field.
    8. That the target field is all numeric and doesnt contain nulls.

    If any of these checks fail, the validator will raise a ValueError.

    Args:
        schema (ForecastingSchema): An instance of ForecastingSchema.
        prediction_field_name (str): Name of the prediction field.

    Returns:
        BaseModel: A dynamic Pydantic BaseModel class for data validation.
    """

    class DataValidator(BaseModel):
        data: pd.DataFrame

        class Config:
            arbitrary_types_allowed = True

        @validator("data", allow_reuse=True)
        def validate_dataframe(cls, data):

            # Check if DataFrame is empty
            if data.empty:
                raise ValueError(
                    "ValueError: The provided predictions file is empty. "
                    "No scores can be generated. "
                )

            id_col = schema.id_col
            time_col = schema.time_col
            # Check that id field is present
            if id_col not in data.columns:
                raise ValueError(
                    "ValueError: Malformed predictions file. "
                    f"ID field '{schema.id_col}' is missing in predictions file."
                )

            # Check for null values in the id column
            if data[id_col].isna().any():
                raise ValueError(
                    f"The ID field '{id_col}' in predictions contains null values. "
                    "Nulls not allowed."
                )

            # Check that time field is present
            if time_col not in data.columns:
                raise ValueError(
                    f"Time field '{time_col}' is missing in predictions file."
                )

            # Check for null values in the time column
            if data[time_col].isna().any():
                raise ValueError(
                    f"The Time field '{time_col}' contains null values. "
                    "Nulls not allowed."
                )

            # Check for duplicates in the combination of id and time col
            if data.duplicated(subset=[id_col, time_col]).any():
                raise ValueError(
                    f"Duplicate entries detected in the data: The combination of the ID field "
                    f"'{id_col}' and the time field '{time_col}' should be unique for each entry. "
                    "Please ensure that each (ID, time) pair is distinct and not repeated."
                )

            # Check that prediction field is present
            if prediction_field_name not in data.columns:
                raise ValueError(
                    "ValueError: Malformed predictions file. "
                    f"Prediction field '{prediction_field_name}' is not present "
                    "in predictions file."
                )

            # Check for null and non-numeric values in the prediction column
            if any(data[prediction_field_name].apply(lambda x: pd.isnull(x) or not np.isreal(x))):
                raise ValueError(
                    f"Target '{prediction_field_name}' contains null or non-numeric data."
                )
            return data

    return DataValidator


def validate_predictions(
    predictions: pd.DataFrame, data_schema: ForecastingSchema, prediction_field_name: str
) -> pd.DataFrame:
    """
    Validates the predictions using the provided schema.

    Args:
        predictions (pd.DataFrame): Predictions data to validate.
        data_schema (ForecastingSchema): An instance of
            ForecastingSchema.
        prediction_field_name (str): Name of the prediction field

    Returns:
        pd.DataFrame: The validated data.
    """
    DataValidator = get_predictions_validator(data_schema, prediction_field_name)
    try:
        validated_data = DataValidator(data=predictions)
        return validated_data.data
    except ValueError as exc:
        raise ValueError(f"Prediction data validation failed: {str(exc)}") from exc


if __name__ == "__main__":
    schema_dict = {
        "title": "Smoke Test Forecasting",
        "description": "This dataset comprises five unique time series with varying components, including sine-wave patterns, linear trends, periodic features, and random noise. It serves as an efficient resource for testing time series forecasting models and exploring pattern recognition and periodicity analysis. ",
        "modelCategory": "forecasting",
        "schemaVersion": 1.0,
        "inputDataFormat": "CSV",
        "encoding": "utf-8",
        "frequency": "OTHER",
        "forecastLength": 3,
        "idField": {
            "name": "series_id",
            "description": "Unique identifier per series"
        },
        "timeField": {
            "name": "t",
            "description": "Time step number of observation.",
            "dataType": "INT",
            "example": 1
        },
        "forecastTarget": {
            "name": "y",
            "description": "Target variable",
            "dataType": "NUMERIC",
            "example": 0.3
        },
        "pastCovariates": [],
        "futureCovariates": [
            {
            "name": "special_event",
            "description": "Numeric variable representing that an offset is added on this sample",
            "dataType": "NUMERIC",
            "example": 0
            }
        ]
    }

    schema_provider = ForecastingSchema(schema_dict)
    predictions = pd.DataFrame(
        {
            "series_id": [1, 1, 1, 2, 2, 2],
            "t": [7, 8, 9, 7, 8, 9],
            "prediction": [0.9, 0.2, 0.8, 0.1, 0.85, 0.2]
        }
    )

    validated_data = validate_predictions(predictions, schema_provider)
