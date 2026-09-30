import sys
from pandas import DataFrame
import pandas as pd
from src.entity.config_entity import IntentPredictorConfig
from src.entity.s3_estimator import Cartpulse_Loaded_Estimator
from src.exception import CustomException
from src.logger import logger

class ProductData:
    """
    Product data constructor.

    Contains all input features required by the trained model
    to predict PurchaseIntent.
    """
    def __init__(
        self,
        ProductCategory,
        ProductBrand,
        ProductPrice,
        CustomerAge,
        CustomerGender,
        PurchaseFrequency,
        CustomerSatisfaction
    ):
        try:
            self.ProductCategory = ProductCategory
            self.ProductBrand = ProductBrand
            self.ProductPrice = ProductPrice
            self.CustomerAge = CustomerAge
            self.CustomerGender = CustomerGender
            self.PurchaseFrequency = PurchaseFrequency
            self.CustomerSatisfaction = CustomerSatisfaction
        except Exception as e:
            logger.error(CustomException(e, sys))

    def get_product_input_data_frame(self) -> DataFrame:
        """
        Create a pandas DataFrame from the product data.
        """
        try:
            product_input_dict = self.get_product_data_as_dict()
            return DataFrame(product_input_dict)
        except Exception as e:
            logger.error(CustomException(e, sys))


    def get_product_data_as_dict(self):
        """
        Convert product data into a dictionary.
        """
        logger.info(
            "Entered get_product_data_as_dict method of ProductData class"
        )
        try:
            input_data = {
                "ProductCategory": [self.ProductCategory],
                "ProductBrand": [self.ProductBrand],
                "ProductPrice": [self.ProductPrice],
                "CustomerAge": [self.CustomerAge],
                "CustomerGender": [self.CustomerGender],
                "PurchaseFrequency": [self.PurchaseFrequency],
                "CustomerSatisfaction": [self.CustomerSatisfaction]
            }

            logger.info("Created product data dictionary")
            logger.info("Exited get_product_data_as_dict method of ProductData class")
            return input_data

        except Exception as e:
            logger.error(CustomException(e, sys))

class ProductDataClassifier:
    def __init__(
        self,
        prediction_pipeline_config: IntentPredictorConfig = IntentPredictorConfig()
    ) -> None:
        """
        Configuration for the prediction pipeline.
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            logger.error(CustomException(e, sys))


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

    def _add_missing_columns(self, df):
        """Add columns that were present during training but are missing during prediction."""
        expected_columns = [
            "Category_Smartphones",
            "Category_Smartwatches",
            "Category_Tablets",
            "Category_Laptops",
            "Category_Headphones",
            "Brand_Others",
            "Brand_Samsung",
            "Brand_Sony",
            "Brand_HP",
            "Brand_Apple"
        ]

        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        return df

    def predict(self, dataframe) -> str:
        """
        Load the trained model and make a prediction.

        Returns:
            Model prediction.
        """
        try:
            logger.info(
                "Entered predict method of ProductDataClassifier class"
            )
            model = Cartpulse_Loaded_Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            dataframe = self._create_dummy_columns(dataframe)
            dataframe = self._rename_columns(dataframe)
            dataframe = self._add_missing_columns(dataframe)
            result = model.predict(dataframe)
            logger.info(
                "Exited predict method of ProductDataClassifier class"
            )
            return result

        except Exception as e:
            logger.error(CustomException(e, sys))

