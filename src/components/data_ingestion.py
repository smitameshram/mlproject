import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig

@dataclass #decorator: used when only variables have to be intialized
class DataIngestionConfig:
    #paths to save train, test, raw data in separate files under folder artifact
    train_data_path: str = os.path.join('artifact', 'train.csv')
    test_data_path: str = os.path.join('artifact', 'test.csv')
    raw_data_path: str = os.path.join('artifact', 'data.csv')

class DataIngestion:
    def __init__ (self):
        #will hold the train, test, raw data path
        self.ingestion_config = DataIngestionConfig()

    def initite_data_ingestion(self):
        #to read data from database/csv
        logging.info("Entered the data ingestion method or component")
        try:
            data_df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Read the dataset as dataframe")

            #create artifact directory and create empty file train.csv, main objective is to create folder artifact
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok= True)
            
            #saves dataframe as csv in the raw data path under artifact folder
            data_df.to_csv(self.ingestion_config.raw_data_path, index= False, header= True)  

            logging.info("Train test split initiated")

            #split raw data as train and test data
            train_set, test_set = train_test_split(data_df, test_size= 0.2, random_state= 42)

            #saves train, test data as csv
            train_set.to_csv(self.ingestion_config.train_data_path, index= False, header= True)  
            test_set.to_csv(self.ingestion_config.test_data_path, index= False, header= True)  

            logging.info("Ingestion of data completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initite_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)