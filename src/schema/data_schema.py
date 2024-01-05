import os
from typing import Dict, List, Tuple

import joblib

from data_models.schema_validator import validate_schema_dict
from utils import read_json_as_dict

SCHEMA_FILE_NAME = "schema.joblib"


class ForecastingSchema:
    """
    A class for loading and providing access to a forecaster schema.

    This class allows users to work with a generic schema for forecaster
    problems, enabling them to create algorithm implementations that are not hardcoded
    to specific feature names. The class provides methods to retrieve information about
    the schema, such as the ID field, target field, time field (if provided), and 
    exogenous fields (if provided). This makes it easier to preprocess and manipulate
    the input data according to the schema, regardless of the specific dataset used.
    """

    def __init__(self, schema_dict: dict) -> None:
        """
        Initializes a new instance of the `ForecastingSchema` class
        and using the schema dictionary.

        Args:
            schema_dict (dict): The python dictionary of schema.
        """
        self.schema = schema_dict
        self._past_covariates = self._get_past_covariates()
        self._future_covariates = self._get_future_covariates()
        self._static_covariates = self._get_static_covariates()

    @property
    def model_category(self) -> str:
        """
        Gets the model category.

        Returns:
            str: The category of the machine learning model
        """
        return self.schema["modelCategory"]

    @property
    def title(self) -> str:
        """
        Gets the title of the dataset or problem.

        Returns:
            str: The title of the dataset or the problem.
        """
        return self.schema["title"]

    @property
    def description(self) -> str:
        """
        Gets the description of the dataset or problem.

        Returns:
            str: A brief description of the dataset or the problem.
        """
        return self.schema["description"]

    @property
    def schema_version(self) -> float:
        """
        Gets the version number of the schema.

        Returns:
            float: The version number of the schema.
        """
        return self.schema["schemaVersion"]

    @property
    def input_data_format(self) -> str:
        """
        Gets the format of the input data.

        Returns:
            str: The format of the input data (e.g., CSV, JSON, etc.).
        """
        return self.schema["inputDataFormat"]

    @property
    def encoding(self) -> str:
        """
        Gets the encoding of the input data.

        Returns:
            str: The encoding of the input data (e.g., "utf-8", "iso-8859-1", etc.).
        """
        return self.schema["encoding"]

    @property
    def frequency(self) -> str:
        """
        Gets the frequency of the data.

        Returns:
            str: The frequency of the day.
        """
        return str(self.schema["frequency"])

    @property
    def forecast_length(self) -> int:
        """
        Gets the forecast_length of the data.

        Returns:
            int: The forecast_length of the data.
        """
        return int(self.schema["forecastLength"])

    @property
    def past_covariates(self) -> List[str]:
        """
        Gets the past_covariates of the data.

        Returns:
            List[str]: The past covariates list.
        """
        return self._past_covariates

    def _get_past_covariates(self) -> List[str]:
        """
        Returns the names of past covariates.

        Returns:
            List[str]: The list of past_covariates.
        """
        if "pastCovariates" not in self.schema:
            return []
        if len(self.schema["pastCovariates"]) == 0:
            return []
        fields = self.schema["pastCovariates"]
        past_covariates = [f["name"] for f in fields if f["dataType"] == "NUMERIC"]
        return past_covariates

    @property
    def future_covariates(self) -> List[str]:
        """
        Gets the future_covariates of the data.

        Returns:
            List[str]: The future covariates list.
        """
        return self._future_covariates

    def _get_future_covariates(self) -> List[str]:
        """
        Returns the names of future covariates.

        Returns:
            List[str]: The list of future_covariates.
        """
        if "futureCovariates" not in self.schema:
            return []
        if len(self.schema["futureCovariates"]) == 0:
            return []
        fields = self.schema["futureCovariates"]
        future_covariates = [f["name"] for f in fields if f["dataType"] == "NUMERIC"]
        return future_covariates

    @property
    def static_covariates(self) -> List[str]:
        """
        Gets the static_covariates of the data.

        Returns:
            List[str]: The static covariates list.
        """
        return self._static_covariates

    def _get_static_covariates(self) -> List[str]:
        """
        Returns the names of static covariates.

        Returns:
            List[str]: The list of static_covariates.
        """
        if "staticCovariates" not in self.schema:
            return []
        if len(self.schema["staticCovariates"]) == 0:
            return []
        fields = self.schema["staticCovariates"]
        static_covariates = [f["name"] for f in fields if f["dataType"] == "NUMERIC"]
        return static_covariates

    @property
    def covariates(self) -> List[str]:
        """
        Gets the past and future covariates of the data.

        Returns:
            List[str]: The covariates list.
        """
        return self._past_covariates + self._future_covariates + self.static_covariates

    @property
    def all_fields(self) -> List[str]:
        """
        Gets the list of all fields in the data.
        This includes the ID, time, target, and covariates.

        Returns:
            List[str]: The list of fields.
        """
        return [self.id_col, self.time_col, self.target] + \
            self.covariates

    @property
    def id_col(self) -> str:
        """
        Gets the name of the ID field.

        Returns:
            str: The name of the ID field.
        """
        return self.schema["idField"]["name"]

    @property
    def id_description(self) -> str:
        """
        Gets the description for the ID field.

        Returns:
            str: The description for the ID field.
        """
        return self.schema["id"].get(
            "description", "No description for target available."
        )

    @property
    def time_col(self) -> str:
        """
        Gets the name of the time field.

        Returns:
            str: The name of the ID field.
        """
        if "timeField" not in self.schema:
            return None
        return self.schema["timeField"]["name"]

    @property
    def time_col_dtype(self) -> str:
        """
        Gets the data type of the time field.

        Returns:
            str: The data type of the ID field.
        """
        if "timeField" not in self.schema:
            return None
        return self.schema["timeField"]["dataType"]

    @property
    def time_description(self) -> str:
        """
        Gets the description for the ID field.

        Returns:
            str: The description for the ID field.
        """
        if "timeField" not in self.schema:
            return "No time field specified in schema"
        return self.schema["timeField"].get(
            "description", "No description for time field available."
        )

    @property
    def target(self) -> str:
        """
        Gets the name of the target field to forecast.

        Returns:
            str: The name of the target field.
        """
        return self.schema["forecastTarget"]["name"]

    @property
    def target_description(self) -> str:
        """
        Gets the description for the target field.

        Returns:
            str: The description for the target field.
        """
        return self.schema["forecastTarget"].get(
            "description", "No description for target available."
        )

    def get_description_for_covariate(self, covariate_name: str) -> str:
        """
        Gets the description for a single covariate.

        Args:
            covariate_name (str): The name of the covariate.

        Returns:
            str: The description for the specified covariate.
        """
        field = self._get_field_by_name(covariate_name)
        return field.get("description", "No description for covariate available.")

    def get_example_value_for_covariate(self, covariate_name: str) -> List[str]:
        """
        Gets the example value for a single covariate.

        Args:
            covariate_name (str): The name of the covariate.

        Returns:
            List[str]: The example values for the specified covariate.
        """
        return self._get_field_by_name(covariate_name).get("example", 0.0)

    def _get_field_by_name(self, covariate_name: str) -> dict:
        """
        Gets the covariate dictionary for a given feature name.

        Args:
            covariate_name (str): The name of the covariate.

        Returns:
            dict: The covariate dictionary for the covariate.

        Raises:
            ValueError: If the covariate is not found in the schema.
        """
        covariates = self.schema["pastCovariates"] \
            + self.schema["futureCovariates"] \
            + self.schema["staticCovariates"]
        for covariate in covariates:
            if covariate["name"] == covariate_name:
                return covariate
        raise ValueError(f"covariate '{covariate_name}' not found in the schema.")


def load_json_data_schema(schema_dir_path: str) -> ForecastingSchema:
    """
    Load the JSON file schema into a dictionary, validate the schema dict for
    its correctness, and use the validated schema to instantiate the schema provider.

    Args:
    - schema_dir_path (str): Path from where to read the schema json file.

    Returns:
        ForecastingSchema: An instance of the ForecastingSchema.
    """
    schema_dict = read_json_as_dict(input_path=schema_dir_path)
    validated_schema_dict = validate_schema_dict(schema_dict=schema_dict)
    data_schema = ForecastingSchema(validated_schema_dict)
    return data_schema


def save_schema(schema: ForecastingSchema, save_dir_path: str) -> None:
    """
    Save the schema to a JSON file.

    Args:
        schema (ForecastingSchema): The schema to be saved.
        save_dir_path (str): The dir path to save the schema to.
    """
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    file_path = os.path.join(save_dir_path, SCHEMA_FILE_NAME)
    joblib.dump(schema, file_path)


def load_saved_schema(save_dir_path: str) -> ForecastingSchema:
    """
    Load the saved schema from a JSON file.

    Args:
        save_dir_path (str): The path to load the schema from.

    Returns:
        ForecastingSchema: An instance of the ForecastingSchema.
    """
    file_path = os.path.join(save_dir_path, SCHEMA_FILE_NAME)
    if not os.path.exists(file_path):
        print("no such file")
        raise FileNotFoundError(f"No such file or directory: '{file_path}'")
    return joblib.load(file_path)
