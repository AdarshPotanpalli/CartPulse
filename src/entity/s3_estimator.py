from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import CustomException
from src.logger import logger
from src.entity.estimator import MyModel
import sys
from pandas import DataFrame


class Cartpulse_Loaded_Estimator:
    """
    This class is used to save and retrieve our model from s3 bucket and to do prediction
    """

    def __init__(self,bucket_name,model_path,):
        """
        :param bucket_name: Name of your model bucket
        :param model_path: Location of your model in bucket
        """
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model:MyModel=None


    def is_model_present(self,model_path):
        """
        Check if the model is available
        """
        return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)

    def load_model(self,)->MyModel:
        """
        Load the model from the model_path
        """
        return self.s3.load_model(self.model_path,bucket_name=self.bucket_name)

    def save_model(self,from_file,remove:bool=False)->None:
        """
        Save(upload) the model to the model_path in S3
        
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        """
        try:
            self.s3.upload_file(from_file,
                                to_filename=self.model_path,
                                bucket_name=self.bucket_name,
                                remove=remove
                                )
        except Exception as e:
            logger.error(CustomException(e))


    def predict(self,dataframe:DataFrame):
        """
        predict on the passed dataframe
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            logger.error(CustomException(e))