import os
import sys
import pymongo
import certifi

from src.exception import CustomException
from src.logger import logger
from src.constants import DATABASE_NAME, MONGODB_URL_KEY

# Load the certificate authority file to avoid timeout errors when connecting to MongoDB
ca = certifi.where()

class MongoDBClient:
    """
    MongoDBClient is responsible for establishing a connection to the MongoDB database.
    """
    client = None  # Shared MongoClient instance across all MongoDBClient instances

    def __init__(self, database_name: str = DATABASE_NAME) -> None:
        """
        Initializes a connection to the MongoDB database. If no existing connection is found, it establishes a new one.
        """
        try:
            # Check if a MongoDB client connection has already been established; if not, create a new one
            if MongoDBClient.client is None:
                mongo_db_url = MONGODB_URL_KEY  # Retrieve MongoDB URL from environment variables                
                # Establish a new MongoDB client connection
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
                
            # Use the shared MongoClient for this instance
            self.client = MongoDBClient.client
            self.database = self.client[database_name]  # Connect to the specified database
            self.database_name = database_name
            logger.info("MongoDB connection successful.")
            
        except Exception as e:
            # Raise a custom exception with traceback details if connection fails
            logger.error(CustomException(e, sys))