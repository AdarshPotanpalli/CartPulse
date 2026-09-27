import sys
from src.exception import CustomException
from src.logger import logger

from src.components.data_ingestion import DataIngestion

from src.entity.config_entity import (DataIngestionConfig)
                                          
from src.entity.artifact_entity import (DataIngestionArtifact)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        try:
            logger.info("Entered the start_data_ingestion method of TrainPipeline class")
            logger.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logger.info("Got the train_set and test_set from mongodb")
            logger.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            logger.error(CustomException(e, sys))

    def run_pipeline(self, ) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
        except Exception as e:
            logger.error(CustomException(e, sys))