import sys

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import CustomException
from src.logger import logger
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import Cartpulse_Loaded_Estimator


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact # needed to decide whether to push model to S3 or not
        self.model_pusher_config = model_pusher_config
        self.cartpulse_loaded_estimator = Cartpulse_Loaded_Estimator(bucket_name=model_pusher_config.bucket_name,
                                model_path=model_pusher_config.s3_model_key_path)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        This function is used to initiate all steps of the model pusher
        
        Output      :   Returns model evaluation artifact
        """
        logger.info("Entered initiate_model_pusher method of ModelTrainer class")

        try:
            print("------------------------------------------------------------------------------------------------")
            # logger.info("Uploading artifacts folder to s3 bucket")
            
            logger.info("Uploading new model to S3 bucket....")
            self.cartpulse_loaded_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)
            model_pusher_artifact = ModelPusherArtifact(bucket_name=self.model_pusher_config.bucket_name,
                                                        s3_model_path=self.model_pusher_config.s3_model_key_path)

            # logger.info("Uploaded artifacts folder to s3 bucket")
            logger.info(f"Model pusher artifact: [{model_pusher_artifact}]")
            logger.info("Exited initiate_model_pusher method of ModelTrainer class")
            
            return model_pusher_artifact
        except Exception as e:
            logger.error(CustomException(e, sys))