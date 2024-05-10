import pandas as pd
import numpy as np
from src.exception import CustomException
from src.components.data_ingestion import DataIngestionConfig 
from src.components.data_ingestion import DataIngestion
from src.logging import logging 
from src.utils.utils import save_model
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
    categorical_cols = ['cut', 'color','clarity']
    numerical_cols = ['carat','depth','table','x','y', 'z','volume','density','table_percentage','depth_percentage','symmetry','surface_area',
 'depth_to_table_ratio','depth_to_diameter_ratio','ddr_refrac_disper','volume_to_surface_area_ratio','vsa_refrac_disper','radius','curv_refrac_disper',
 'thickness','thick_refrac_disper']
            
    cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
    color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']



    def __init__(self):
        self.TransformationConfig=DataTransformationConfig()

    def new_features(self,df):
        logging.info("Started calculating new features")
        df['x'] = df['x'].replace(0, 0.5)
        df['y'] = df['y'].replace(0, 0.5)
        df['z'] = df['z'].replace(0, 0.5)
        df['volume'] = df['x'] * df['y'] * df['z']
        df['density'] = df['carat'] / df['volume']
        df['table_percentage'] = (df['table'] / ((df['x'] + df['y']) / 2)) * 100
        df['depth_percentage'] = (df['depth'] / ((df['x'] + df['y']) / 2)) * 100
        df['symmetry'] = (abs(df['x'] - df['z']) + abs(df['y'] - df['z'])) / (df['x'] + df['y'] + df['z'])
        df['surface_area'] = 2 * ((df['x'] * df['y']) + (df['x'] * df['z']) + (df['y'] * df['z']))
        df['depth_to_table_ratio'] = df['depth'] / df['table']
        logging.info("Calculated new features")
        
        return df
    
    def new_features1(self,df):
        logging.info("Started calculating new features 1")
        avg_refrac_index = 2.165 
        avg_disper_value = 0.062 

        df['depth_to_diameter_ratio'] = df['depth'] / ((df['x'] + df['y'] + 1e-6) / 2)
        #extending to:
        df['ddr_refrac_disper'] = df['depth_to_diameter_ratio'] * avg_refrac_index * avg_disper_value

        df['volume_to_surface_area_ratio'] = df['carat'] / df['table']
        df['vsa_refrac_disper'] = df['volume_to_surface_area_ratio'] * avg_refrac_index * avg_disper_value

        df['radius'] = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2)
        df['curv_refrac_disper'] = (avg_refrac_index * avg_disper_value) / df['radius'] 

        df['thickness'] = df['z'] - ((df['x'] + df['y'])/2)
        df['thick_refrac_disper'] = df['thickness'] * avg_refrac_index * avg_disper_value  

        return df
    
    #def Encoding(self,df)

   
    def get_data_transformation(self):
        try:
            logging.info('Initiated transformation')

            logging.info('Pipeline Initiated')
            
            # Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]

            )
            
            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[self.cut_categories,self.color_categories,self.clarity_categories])),
                ('scaler',StandardScaler())
                ]

            )
            
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,self.numerical_cols),
                ('cat_pipeline',cat_pipeline,self.categorical_cols)
            ])
            
            return preprocessor


        except Exception as e:
            logging.info("Exception occured during get_data_transformation ")
            
            raise CustomException(e,sys)
    


    def initiate_data_transformation(self,train_path,test_path):
        try:
                
            train=pd.read_csv(train_path)
            test=pd.read_csv(test_path)

            logging.info("Read training and testing datasets")
            logging.info(f"Training dataset:\n{train.head().to_string()}")
            logging.info(f"Testing dataset:\n{test.head().to_string()}")

            preprocessing_obj=self.get_data_transformation()

            target_col_name = "price"
            drop_col=[target_col_name,'id']

            input_feature_train = train.drop(columns=drop_col,axis=1)
            target_feature_train=train[target_col_name]
            
            
            input_feature_test=test.drop(columns=drop_col,axis=1)
            logging.info(f"shape of input feature test set:{input_feature_test.shape}")
            target_feature_test=test[target_col_name]
            

            new_features_train=self.new_features(input_feature_train)
            logging.info(f"new fetaures added:\n{new_features_train.head().to_string()}")
            new_features_train=self.new_features1(new_features_train)
            logging.info(f"Some other new fetaures added:\n{new_features_train.head().to_string()}")
            #new_features=pd.DataFrame(new_features,columns=new_features.columns)

            new_features_test=self.new_features(input_feature_test)
            logging.info(f"new fetaures added to test:\n{new_features_test.head().to_string()}")
            new_features_test=self.new_features1(new_features_test)
            logging.info(f"Some other new fetaures added to test:\n{new_features_test.head().to_string()}")

            input_feature_train_arr=preprocessing_obj.fit_transform(new_features_train)
            logging.info(f"after preprocessing training\n{input_feature_train_arr}")
            logging.info(f"after preprocessing training\n{input_feature_train_arr.shape}")
            input_feature_test_arr=preprocessing_obj.transform(new_features_test)
            logging.info(f"traget training features shape\n{target_feature_train.shape}")
            logging.info(f"traget testing features shape\n{target_feature_test.shape}")
            logging.info(f"after preprocessing testing\n{input_feature_test_arr.shape}")            
            logging.info("Applying preprocessing object on training and testing datasets.")
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train)]
            logging.info(f"after preprocessing testing\n{train_arr}") 
            logging.info(f"after preprocessing testing\n{train_arr.shape}") 
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test)]

            save_model(
                file_path=self.TransformationConfig.preprocessor_obj_file,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")
            
            return (
                train_arr,
                test_arr
            )
            
        except Exception as e:
            logging.info('Error occured during initaiate_data_transformation')

            raise CustomException(e,sys)




""""
if __name__=="__main__":

    obj=DataIngestion()
    train_data_path,test_data_path=obj.initaiate_data_ingestion()

    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data_path,test_data_path)



"""