import pandas as pd
import numpy as np
from src.exception import CustomException
from src.components.data_ingestion import DataIngestionConfig 
from src.components.data_ingestion import DataIngestion
from src.logging import logging 

import os
import sys
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from pathlib import Path
from dataclasses import dataclass

@dataclass
class DataTransformationConfig():
    preprocessor_obj_file=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation():
    def __init__(self):
        self.TransformationConfig=DataTransformationConfig()

    def get_data_transformation(self):
        pass

    def initaiate_data_transformation(self,train_path,test_path):
        train=pd.read_csv(train_path)
        test=pd.read_csv(test_path)

        logging.info("Read training and testing datasets")
        logging.info(f"Training dataset:{train.head().to_string()}")
        logging.info(f"Testing dataset:{train.head().to_string()}")

        preprocessing_obj=self.get_data_transformation()

        target_col_name = "price"
        drop_col=[target_col_name,'id']

        input_feature_train = train_df.drop(columns=drop_columns,axis=1)
        target_feature_train_df=train_df[target_column_name]
        
        
        input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
        target_feature_test_df=test_df[target_column_name]
        





        print(train.head().to_string())
        print(test.head().to_string())
            


if __name__=="__main__":

    obj=DataIngestion()
    train_data_path,test_data_path=obj.initaiate_data_ingestion()

    
    DataTransformation.initaiate_data_transformation(train_data_path,test_data_path)



"""
        except Exception as e:
            logging.info("Error Occured in data Transformation cla")
            raise CustomException(e,sys)  """