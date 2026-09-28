import json
import sys
import os

import pandas as pd

from pandas import DataFrame

from src.exception import CustomException
from src.logger import logger
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact # artifacts from data ingestion are needed during validation
            self.data_validation_config = data_validation_config
            self._schema_config =read_yaml_file(file_path=SCHEMA_FILE_PATH) # creates a dict of schema
        except Exception as e:
            logger.error(CustomException(e, sys))

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        This method validates the number of columns
        Output      :   Returns bool value based on validation results
        """
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logger.info(f"Is required column present: [{status}]")
            return status
        except Exception as e:
            logger.error(CustomException(e, sys))

    def is_column_exist(self, df: DataFrame) -> bool:
        """
        This method validates the existence of a numerical and categorical columns
        Output      :   Returns bool value based on validation results
        """
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns)>0: # checks if the numerical columns are missing
                logger.info(f"Missing numerical column: {missing_numerical_columns}")

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns)>0: # Checks if any categorical columns are missing
                logger.info(f"Missing categorical column: {missing_categorical_columns}")

            return False if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0 else True # whether all specified numerical columns exist or not
        except Exception as e:
            logger.error(CustomException(e, sys))

    @staticmethod # can access this method without creating object
    def read_data(file_path) -> DataFrame:
        """
        Reads csv data in the form of dataframe
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(CustomException(e, sys))
        

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        This method initiates the data validation component for the pipeline
        Output      :   Returns bool value based on validation results
        """

        try:
            validation_error_msg = ""
            logger.info("Starting data validation")
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                                 DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))

            # 1. Checking col len of dataframe for train/test df
            status = self.validate_number_of_columns(dataframe=train_df)
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe. "
            else:
                logger.info(f"All required columns present in training dataframe: {status}")


            status = self.validate_number_of_columns(dataframe=test_df)
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe. "
            else:
                logger.info(f"All required columns present in testing dataframe: {status}")

            # 2. Validating missing columns in the train/test df
            status = self.is_column_exist(df=train_df)
            if not status:
                validation_error_msg += f"Columns are missing in training dataframe. "
            else:
                logger.info(f"All categorical/int columns present in training dataframe: {status}")

            status = self.is_column_exist(df=test_df)
            if not status:
                validation_error_msg += f"Columns are missing in test dataframe."
            else:
                logger.info(f"All categorical/int columns present in testing dataframe: {status}")

            validation_status = len(validation_error_msg) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            # Ensure the directory for validation_report_file_path exists
            report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            # Save validation status and message to a JSON file, which will be an artifact for validation
            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_msg.strip()
            }
            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
                json.dump(validation_report, report_file, indent=4)

            logger.info("Data validation artifact created and saved to JSON file.")
            logger.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            logger.error(CustomException(e, sys))