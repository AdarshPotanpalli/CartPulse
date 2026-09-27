import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import CustomException
from src.logger import logger

class Proj1Data:
    """
    A class to export MongoDB records as a pandas DataFrame.
    """
    def __init__(self) -> None:
        """
        Initializes the MongoDB client connection.
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            logger.error(CustomException(e, sys))

    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        """
        Exports an entire MongoDB collection as a pandas DataFrame.
        """
        try:
            # Access specified collection from the default or specified database
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]

            # Convert collection data to DataFrame and preprocess
            logger.info("Fetching data from mongoDB")
            df = pd.DataFrame(list(collection.find()))
            logger.info(f"Data fecthed with len: {len(df)}")
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"])
            df.replace({"na":np.nan},inplace=True)
            return df

        except Exception as e:
            logger.error(CustomException(e, sys))