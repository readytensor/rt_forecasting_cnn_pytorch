import numpy as np
import pandas as pd
from pydantic import BaseModel, validator

from schema.data_schema import ForecastingSchema
from data_models.schema_validator import TimeDataType


def get_data_validator(schema: ForecastingSchema, is_train: bool) -> BaseModel:
    """
    Returns a dynamic Pydantic data validator class based on the provided schema.

    The resulting validator checks the following:

    1. That the input DataFrame contains the ID field specified in the schema.
    2. That the data under ID field is unique.
    3. If `is_train` is `True`, that the input DataFrame contains the target field
        specified in the schema.
    6. That the input DataFrame contains all feature fields specified in the schema.
    7. For non-nullable features, that they do not contain null values.
    8. That numeric features do not contain non-numeric values.

    If any of these checks fail, the validator will raise a ValueError.

    Args:
        schema (ForecastingSchema): An instance of ForecastingSchema.
        is_train (bool): Whether the data is for training or not. Determines whether
            the presence of a target field is required in the data.

    Returns:
        BaseModel: A dynamic Pydantic BaseModel class for data validation.
    """

    class DataValidator(BaseModel):
        data: pd.DataFrame

        class Config:
            arbitrary_types_allowed = True

        @validator("data", allow_reuse=True)
        def validate_dataframe(cls, data):
            id_col = schema.id_col
            time_col = schema.time_col
            target = schema.target
            time_col_dtype = schema.time_col_dtype

            if id_col not in data.columns:
                raise ValueError(
                    f"ID field '{id_col}' is not present in the given data"
                )

            # Check for null values in the 'id' column
            if data[id_col].isna().any():
                raise ValueError(
                    f"The ID field '{id_col}' contains null values."
                )

            if time_col not in data.columns:
                raise ValueError(
                    f"Time field '{time_col}' is not present in the given data"
                )

            # Check for null values in the 'id' column
            if data[time_col].isna().any():
                raise ValueError(
                    f"The Time field '{time_col}' contains null values."
                )

            # Check if all time column values are integers
            if time_col_dtype == TimeDataType.INT:
                # Check if all time column values are integers
                if not pd.api.types.is_integer_dtype(data[time_col]):
                    raise ValueError(
                        f"The time column '{time_col}' must contain only integer values. "
                        "Found non-integer value(s)."
                    )

            elif time_col_dtype == TimeDataType.DATE:
                # Check if all values match the DATE format '%Y-%m-%d'
                if not all(
                    pd.to_datetime(data[time_col], format='%Y-%m-%d', errors='coerce').notna()
                ):
                    raise ValueError(
                        f"The time column '{time_col}' must be in '%Y-%m-%d' format."
                    )

            elif time_col_dtype == TimeDataType.DATETIME:
                # Check if all values match either of the DATETIME formats
                if not all(
                    pd.to_datetime(data[time_col], format='%Y-%m-%d %H:%M:%S', errors='coerce').notna() |
                    pd.to_datetime(data[time_col], format='%Y-%m-%d %H-:M:%S.%f', errors='coerce').notna()
                ):
                    raise ValueError(
                        f"The time column '{time_col}' must be in '%Y-%m-%d %H:%M:%S'"
                        " or '%Y-%m-%d %H:%M:%S.%f' format."
                    )
            else:
                raise ValueError(f"Unsupported time column data type: '{time_col_dtype}'")

            # Check for duplicates in the combination of id and time col
            if data.duplicated(subset=[id_col, time_col]).any():
                raise ValueError(
                    f"The combination of ID field '{id_col}' and time field "
                    f"'{time_col}' does not contain unique values."
                )

            # Check if all 'id's have the same number of unique 'time' values
            unique_time_counts = data.groupby(id_col)[time_col].nunique()
            if unique_time_counts.nunique() != 1:
                raise ValueError(
                    f"Not all IDs in field '{id_col}' have the same number of unique "
                    f"time values in field '{time_col}'."
                )

            if is_train:
                if target not in data.columns:
                    raise ValueError(
                        f"Target field '{target}' is missing "
                        "in the given data"
                    )

                if not all(data[target].apply(lambda x: pd.isnull(x) or np.isreal(x))):
                    raise ValueError(
                        f"Target '{target}' contains null or non-numeric data"
                    )

                for feature in schema.past_covariates:
                    if feature not in data.columns:
                        raise ValueError(
                            f"Past covariate '{feature}' is missing in the given data"
                        )

            for feature in schema.future_covariates:
                if feature not in data.columns:
                    raise ValueError(
                        f"Future covariate '{feature}' is missing in the given data"
                    )

            covariates_to_check = [*schema.future_covariates]
            if is_train:
                covariates_to_check += schema.past_covariates

            for feature in covariates_to_check:
                if any(data[feature].apply(lambda x: pd.isnull(x) or not np.isreal(x))):
                    raise ValueError(
                        f"Covariate '{feature}' contains non-numeric data. "
                        "Only numeric values allowed."
                    )

            return data

    return DataValidator


def validate_data(
    data: pd.DataFrame, data_schema: ForecastingSchema, is_train: bool
) -> pd.DataFrame:
    """
    Validates the data using the provided schema.

    Args:
        data (pd.DataFrame): The train or test data to validate.
        data_schema (ForecastingSchema): An instance of
            ForecastingSchema.
        is_train (bool): Whether the data is for training or not.

    Returns:
        pd.DataFrame: The validated data.
    """
    DataValidator = get_data_validator(data_schema, is_train)
    try:
        validated_data = DataValidator(data=data)
        return validated_data.data
    except ValueError as exc:
        raise ValueError(f"Data validation failed: {str(exc)}") from exc
