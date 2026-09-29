import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import CustomException
from src.logger import logger
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact # artifacts from ingestion and validation are needed
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            logger.error(CustomException(e, sys))

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(CustomException(e, sys))

    def _scaling_pipeline(self) -> Pipeline:
        """
        Returns a scaling pipeline to scale the columns
        """
        logger.info("Entered _scaling_pipeline method of DataTransformation class")

        try:
            # Initialize transformers
            numeric_transformer = StandardScaler()
            logger.info("Transformers Initialized: StandardScaler")

            # Load schema configurations
            num_features = self._schema_config['num_features']
            logger.info("Cols to be scaled loaded from schema.")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer( # can add min max scaler in future
                transformers=[
                    ("StandardScaler", numeric_transformer, num_features)
                ],
                remainder='passthrough'  # Leaves other columns as they are
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logger.info("Final Pipeline Ready!!")
            logger.info("Exited _scaling_pipeline method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logger.error(CustomException(e, sys))

    def _create_dummy_columns(self, df):
        """Create dummy variables for categorical features."""
        logger.info("Creating dummy variables for categorical features")
        try:
            categorical_cols = self._schema_config['categorical_columns']
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
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
        drop_col = self._schema_config['drop_columns']
        if drop_col in df.columns:
            df = df.drop(columns= [drop_col])
        return df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logger.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                logger.error(self.data_validation_artifact.message)

            # Load train and test data (thats why ingestion artifact is needed)
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logger.info("Train-Test data loaded")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]
            logger.info("Input and Target cols defined for both train and test df.")

            # Apply custom transformations in specified sequence
            # logger.info(input_feature_train_df.head(2))
            input_feature_train_df = self._drop_id_column(input_feature_train_df)
            input_feature_train_df = self._create_dummy_columns(input_feature_train_df)
            input_feature_train_df = self._rename_columns(input_feature_train_df)

            input_feature_test_df = self._drop_id_column(input_feature_test_df)
            input_feature_test_df = self._create_dummy_columns(input_feature_test_df)
            input_feature_test_df = self._rename_columns(input_feature_test_df)
            logger.info("Custom transformations applied to train and test data")

            logger.info("Starting data scaling")
            preprocessor = self._scaling_pipeline()
            logger.info("Got the preprocessor object")

            logger.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logger.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logger.info("Transformation done end to end to train-test df.")

            # creates synthetic samples for minority class
            logger.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            # # SMOTEEN should not be applied to test data
            # input_feature_test_final, target_feature_test_final = smt.fit_resample(
            #     input_feature_test_arr, target_feature_test_df
            # )
            logger.info("SMOTEENN applied to train df only.")

            # column wise concatenation of input and target features
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logger.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor) # saving the scaling pipeline
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logger.info("Saving transformation object and transformed files.")

            logger.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            logger.error(CustomException(e, sys))