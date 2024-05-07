import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logging import logging 
import os
import sys
from sklearn.model_selection import train_test_split
from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path=os.path.join('artifacts','data.csv')
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestionConfig=DataIngestionConfig()

    def initaiate_data_ingestion(self):
        logging.info("Data_ingestion initiated")
        try:
            data=pd.read_csv('data/train.csv')
            logging.info('Read the data successfully')

            os.makedirs(os.path.dirname(os.path.join(self.ingestionConfig.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestionConfig.train_data_path,index=False)
            logging.info('Saved data to artifacts folder')

            train_data,test_data=train_test_split(data,test_size=0.25,random_state=13)
            logging.info('Performed train and test split')

            print(train_data)

            train_data.to_csv(self.ingestionConfig.train_data_path,index=False)
            test_data.to_csv(self.ingestionConfig.test_data_path,index=False)

            logging.info('data ingestion completed')

            return(
                self.ingestionConfig.train_data_path,
                self.ingestionConfig.test_data_path

            )


        except Exception as e:
           logging.info("exception during occured at data ingestion stage")
           raise CustomException(e,sys)


if __name__=="__main__":
    obj=DataIngestion()
    data=obj.initaiate_data_ingestion()