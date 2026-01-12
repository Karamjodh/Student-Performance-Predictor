import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from src.Exception import CustomException
from src.Logger import logging
from src.Utils import save_object
import dataclasses

@ dataclasses.dataclass
class Datatransformationconfig:
    preprocessor_obj_file_path = os.path.join("artifact","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.datatransformation_config = Datatransformationconfig()
    
    def get_transformer_object(self):

        """This function is responsible for data transformation"""

        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            num_pipeline = Pipeline(
                steps = [("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean = False))]
            )
            cat_pipeline = Pipeline(
                steps = [("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown = "ignore")),
                ("Standard_scaler", StandardScaler(with_mean = False))]
            )

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns standard encoding completed")

            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_tranformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtainig preprocessor object")

            preprocessing_object = self.get_transformer_object()
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns = [target_column_name],axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name],axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on train and test data")

            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(file_path = self.datatransformation_config.preprocessor_obj_file_path,
                        obj = preprocessing_object)
            
            return (train_arr, test_arr,self.datatransformation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataTransformation()
    train_data, test_data, preprocessor_path = obj.initiate_data_tranformation("artifact/train.csv","artifact/test.csv")