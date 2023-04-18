import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.utils import save_object
from src.logger import logging
import os

@dataclass
class DataTranformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTranformation:
    def __init__(self):
        self.data_tranformation_config = DataTranformationConfig()

    def get_data_tranformer_object(self):
        '''Responsible for data tranformation'''
        try:
            numerical_columns= ['writing_score','reading_score']
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",]
            num_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_coder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info(f'Categorical colums : {categorical_columns}')
            logging.info(f'Numerical colums : {numerical_columns}')


            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
            ])

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_tranformation(self,train_path,test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info('obtaining preprocessing object')

            preprocesing_object = self.get_data_tranformer_object()

            target_column_name = 'math_score'
            numerical_columns= ['writing_score','reading_score']
            input_features_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_features_train_df= train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_features_test_df= test_df[target_column_name]

            logging.info('Applying preprocessing object on training and testing')

            input_features_train_arr = preprocesing_object.fit_transform(input_features_train_df)
            input_features_test_arr = preprocesing_object.transform(input_features_test_df)

            train_arr = np.c_[input_features_train_arr,np.array(target_features_train_df)]
            test_arr = np.c_[input_features_test_arr,np.array(target_features_test_df)]

            logging.info('Saved preprocessing object')

            save_object(
                file_path = self.data_tranformation_config.preprocessor_obj_file_path,
                obj= preprocesing_object
            )
        
            return (
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path
            )

        except  Exception as e:
            raise CustomException(e,sys)
            
