from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from src.exception import CustomException
from src.constants import TARGET_COLUMN
from src.logger import logger
from src.utils.main_utils import load_object
import sys
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import Cartpulse_Loaded_Estimator
from dataclasses import dataclass

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config 
            self.data_ingestion_artifact = data_ingestion_artifact # test data artifact is needed
            self.model_trainer_artifact = model_trainer_artifact # trained model artifact is needed
        except Exception as e:
            logger.error(CustomException(e))

    def get_best_model(self) -> Optional[Cartpulse_Loaded_Estimator]:
        """
        This function is used to get model from production stage.
        
        Output      :   Returns model object if available in s3 storage
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            cartpulse_loaded_estimator = Cartpulse_Loaded_Estimator(bucket_name=bucket_name,
                                                        model_path=model_path)

            if cartpulse_loaded_estimator.is_model_present(model_path=model_path):
                return cartpulse_loaded_estimator
            return None
        except Exception as e:
            logger.error(CustomException(e))
        
    def _create_dummy_columns(self, df):
        """Create dummy variables for categorical features."""
        logger.info("Creating dummy variables for categorical features")
        try:
            df = pd.get_dummies(df, drop_first=True)
            return df
        except Exception as e:
            logger.error(CustomException(e, sys))

    def _rename_columns(self, df):
        """Rename specific columns and ensure integer types for dummy columns."""
        logger.info("Renaming specific columns and casting to int")
        df = df.rename(columns={
            "ProductCategory_Smartphones": "Category_Smartphones",
            "ProductCategory_Smart Watches": "Category_Smartwatches",
            "ProductCategory_Tablets": "Category_Tablets",
            "ProductCategory_Laptops": "Category_Laptops",
            "ProductCategory_Headphones": "Category_Headphones",
            "ProductBrand_Other Brands": "Brand_Others",
            "ProductBrand_Samsung": "Brand_Samsung",
            "ProductBrand_Sony": "Brand_Sony",
            "ProductBrand_HP": "Brand_HP",
            "ProductBrand_Apple": "Brand_Apple"
        })
        for col in ["Category_Smartphones", "Category_Smartwatches", "Category_Tablets", "Category_Laptops", "Category_Headphones",
                    "Brand_Others", "Brand_Samsung", "Brand_Sony", "Brand_HP", "Brand_Apple"]:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df
    
    def _drop_id_column(self, df):
        """Drop the 'id' column if it exists."""
        logger.info("Dropping 'id' column")
        drop_col = "ProductID"
        if drop_col in df.columns:
            df = df.drop(columns= [drop_col])
        return df

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        This function is used to evaluate trained model with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logger.info("Test data loaded and now transforming it for prediction...")

            x = self._drop_id_column(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)

            # trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logger.info("Trained model loaded/exists.")
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logger.info(f"F1_Score for this model: {trained_model_f1_score}")

            best_model_f1_score=None
            best_model = self.get_best_model()
            if best_model is not None:
                logger.info(f"Computing F1_Score for production model..")
                y_hat_best_model = best_model.predict(x) # applies scaling transformation and then predicts
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logger.info(f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}")
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           ) # a dataclass
            logger.info(f"Result: {result}")
            return result

        except Exception as e:
            logger.error(CustomException(e, sys))

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        """  
        try:
            print("------------------------------------------------------------------------------------------------")
            logger.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logger.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            logger.error(CustomException(e, sys))