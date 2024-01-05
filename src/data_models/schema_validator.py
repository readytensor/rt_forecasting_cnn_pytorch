from collections import Counter
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, ValidationError, validator


class ID(BaseModel):
    """
    A model representing the ID field of the dataset.
    """
    name: str
    description: str


class ForecastTarget(BaseModel):
    """
    A model representing the target field of forecasting problem.
    """
    name: str
    description: str
    dataType: str
    example: float


class TimeDataType(str, Enum):
    """Enum for the data type of a feature"""

    DATE = "DATE"
    DATETIME = "DATETIME"
    INT = "INT"


class TimeField(BaseModel):
    """
    A model representing the target field of forecasting problem.
    """
    name: str
    description: str
    dataType: TimeDataType
    example: Union[float, str]


class CovariateDataType(str, Enum):
    """Enum for the data type of a Covariate"""

    NUMERIC = "NUMERIC"
    INT = "INT"
    FLOAT = "FLOAT"


class Frequency(str, Enum):
    """Enum for the frequency of the time series"""

    SECONDLY = "SECONDLY"
    MINUTELY = "MINUTELY"
    HOURLY = "HOURLY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    YEARLY = "YEARLY"
    OTHER = "OTHER"

class Covariate(BaseModel):
    """
    A model representing the predictor fields in the dataset. Validates the
    presence and type of the 'example' field based on the 'dataType' field
    for NUMERIC dataType and presence and contents of the 'categories' field
    for CATEGORICAL dataType.
    """

    name: str
    description: str
    dataType: CovariateDataType
    example: Optional[float]

class SchemaModel(BaseModel):
    """
    A schema validator for binary classification problems. Validates the
    problem category, version, and predictor fields of the input schema.
    """

    title: str
    description: str = None
    modelCategory: str
    schemaVersion: float
    inputDataFormat: str = None
    encoding: str = None
    frequency: Frequency
    forecastLength: int
    idField: ID
    timeField: TimeField
    forecastTarget: ForecastTarget
    pastCovariates: List[Covariate]
    futureCovariates: List[Covariate]
    staticCovariates: List[Covariate]

    @validator("modelCategory", allow_reuse=True)
    def valid_problem_category(cls, v):
        if v != "forecasting":
            raise ValueError(
                f"modelCategory must be 'forecasting'. Given {v}"
            )
        return v

    @validator("schemaVersion", allow_reuse=True)
    def valid_version(cls, v):
        if v != 1.0:
            raise ValueError(f"schemaVersion must be set to 1.0. Given {v}")
        return v

    @validator("pastCovariates", allow_reuse=True)
    def unique_past_covariate_names(cls, v):
        """
        Check that the past covariates names are unique.
        """
        feature_names = [feature.name for feature in v]
        duplicates = [
            item for item, count in Counter(feature_names).items() if count > 1
        ]

        if duplicates:
            raise ValueError(
                "Duplicate past covariates names found in schema: " f"`{', '.join(duplicates)}`"
            )

        return v

    @validator("futureCovariates", allow_reuse=True)
    def unique_future_covariate_names(cls, v):
        """
        Check that the future covariates names are unique.
        """
        feature_names = [feature.name for feature in v]
        duplicates = [
            item for item, count in Counter(feature_names).items() if count > 1
        ]

        if duplicates:
            raise ValueError(
                "Duplicate future covariates names found in schema: " f"`{', '.join(duplicates)}`"
            )

        return v


def validate_schema_dict(schema_dict: dict) -> dict:
    """
    Validate the schema
    Args:
        schema_dict: dict
            data schema as a python dictionary

    Raises:
        ValueError: if the schema is invalid

    Returns:
        dict: validated schema as a python dictionary
    """
    try:
        schema_dict = SchemaModel.parse_obj(schema_dict).dict()
        return schema_dict
    except ValidationError as exc:
        raise ValueError(f"Invalid schema: {exc}") from exc
