import os
import sys
import pickle
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logging import logging

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


def save_model(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        logging.info('Exception Occured in save_model function utils')
        CustomException(e,sys)


def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)