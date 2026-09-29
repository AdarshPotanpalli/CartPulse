import sys
from typing import Tuple

import numpy as np
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import CustomException
from src.logger import logger
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        This function trains an XGBoost classifier with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        """
        try:
            logger.info("Training XGBoost classifier with specified parameters")

            # Splitting the train and test data into features and target variables
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logger.info("train-test split input and target seperation done.")

            # Initialize RandomForestClassifier with specified parameters
            model = XGBClassifier(
                n_estimators=self.model_trainer_config._n_estimators,
                max_depth=self.model_trainer_config._max_depth,
                learning_rate=self.model_trainer_config._learning_rate,
                subsample=self.model_trainer_config._subsample,
                colsample_bytree=self.model_trainer_config._colsample_bytree,
                min_child_weight=self.model_trainer_config._min_child_weight,
                gamma=self.model_trainer_config._gamma,
                random_state=self.model_trainer_config._random_state,
                eval_metric="logloss"
            )

            # Fit the model
            logger.info("Model training going on...")
            model.fit(x_train, y_train)
            logger.info("Model training done.")

            # Predictions and evaluation metrics
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            # Creating metric artifact
            metric_artifact = ClassificationMetricArtifact(accuracy= accuracy, f1_score=f1, precision_score=precision, recall_score=recall)
            return model, metric_artifact
        
        except Exception as e:
            logger.error(CustomException(e, sys))

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        """
        logger.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            logger.info("Starting Model Trainer Component")
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logger.info("train-test data loaded")
            
            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logger.info("Model object and artifact loaded.")
            
            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logger.info("Preprocessing obj loaded.")

            # Check if the model's accuracy meets the expected threshold
            if accuracy_score(train_arr[:, -1], trained_model.predict(train_arr[:, :-1])) < self.model_trainer_config.expected_accuracy:
                logger.info("No model found with score above the base score")
                raise Exception("No model found with score above the base score")

            # Save the final model object that includes both preprocessing and the trained model
            logger.info("Saving new model as performace is better than expected accuracy.")
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model) # here my model is only used to save the trained model and preprocessing as a single artifact file
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logger.info("Saved final model object that includes both preprocessing and the trained model")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logger.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            logger.error(CustomException(e, sys))