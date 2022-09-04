from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import datetime as dt

import pandas as pd
import numpy as np

##############################################################################################################################
# 작업에서 사용하는 함수
##############################################################################################################################

def preprocess_dataset(initial_dataset: pd.DataFrame, date: dt.datetime="None"):
    """이 함수는 모델에서 사용할 데이터셋을 사전 처리합니다.

    Args:
        initial_dataset (pd.DataFrame): 데이터를 처음 읽을 때의 원시 형식

    Returns:
        pd.DataFrame: 분류를 위해 사전 처리된 데이터 세트
    """
    print("\n     데이터세트 전처리 중...")
    
    #날짜의 데이터 프레임을 필터링합니다.
    if date != "None":
        initial_dataset['Date'] = pd.to_datetime(initial_dataset['Date'])
        processed_dataset = initial_dataset[initial_dataset['Date'] <= date]
        print(len(processed_dataset))
    else:
        processed_dataset = initial_dataset
        
    processed_dataset = processed_dataset[['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Exited']]
    
    
    processed_dataset = pd.get_dummies(processed_dataset)

    if 'Gender_Female' in processed_dataset.columns:
        processed_dataset.drop('Gender_Female',axis=1,inplace=True)
        
    processed_dataset = processed_dataset.apply(pd.to_numeric)
    
    columns_to_select = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                         'IsActiveMember', 'EstimatedSalary',  'Geography_France', 'Geography_Germany',
                         'Geography_Spain',  'Gender_Male','Exited']
    
    processed_dataset = processed_dataset[[col for col in columns_to_select if col in processed_dataset.columns]]

    print("     전처리 완료!\n")
    return processed_dataset


def create_train_test_data(preprocessed_dataset: pd.DataFrame):
    """이 함수는 데이터 세트를 분할하여 기차 데이터를 생성합니다.

    Args:
        preprocessed_dataset (pd.DataFrame): 전처리된 데이터 세트

    Returns:
        pd.DataFrame: 훈련 데이터 세트
    """
    print("\n     Creating the training and testing dataset...")
    
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_dataset.iloc[:,:-1],preprocessed_dataset.iloc[:,-1],test_size=0.2,random_state=42)
    
    train_data = pd.concat([X_train,y_train],axis=1)
    test_data = pd.concat([X_test,y_test],axis=1)
    print("     생성 완료!")
    return train_data, test_data


def train_model_baseline(train_dataset: pd.DataFrame):
    """로지스틱 회귀 모델을 훈련시키는 함수

    Args:
        train_dataset (pd.DataFrame): 학습 데이터세트

    Returns:
        model (LogisticRegression): 피팅된 모델
    """
    print("     모델 학습 중...\n")
    X,y = train_dataset.iloc[:,:-1],train_dataset.iloc[:,-1]
    model_fitted = LogisticRegression().fit(X,y)
    print("\n    ",model_fitted," 학습되었습니다!")
    
    importance_dict = {'Features' : X.columns, 'Importance':model_fitted.coef_[0]}
    importance = pd.DataFrame(importance_dict).sort_values(by='Importance',ascending=True)
    return model_fitted, importance

def train_model(train_dataset: pd.DataFrame):
    """로지스틱 회귀 모델을 훈련시키는 함수

    Args:
        train_dataset (pd.DataFrame): 학습 데이터

    Returns:
        model (RandomForest): 피팅된 모델
    """
    print("     모델 학습 중...\n")
    X,y = train_dataset.iloc[:,:-1],train_dataset.iloc[:,-1]
    model_fitted = RandomForestClassifier().fit(X,y)
    print("\n    ",model_fitted," 학습되었습니다!")
    
    importance_dict = {'Features' : X.columns, 'Importance':model_fitted.feature_importances_}
    importance = pd.DataFrame(importance_dict).sort_values(by='Importance',ascending=True)
    return model_fitted, importance

def forecast(test_dataset: pd.DataFrame, trained_model: RandomForestClassifier):
    """테스트 데이터 세트를 예측하는 기능

    Args:
        test_dataset (pd.DataFrame): 테스트 데이터 세트
        trained_model (LogisticRegression): 피팅된 모델

    Returns:
        forecast (pd.DataFrame): 예측 데이터 세트
    """
    print("     테스트 데이터 세트 예측 중...")
    X,y = test_dataset.iloc[:,:-1],test_dataset.iloc[:,-1]
    #predictions = trained_model.predict(X)
    predictions = trained_model.predict_proba(X)[:, 1]
    print("     예측 완료!")
    return predictions


def forecast_baseline(test_dataset: pd.DataFrame, trained_model: LogisticRegression):
    """테스트 데이터 세트를 예측하는 기능

    Args:
        test_dataset (pd.DataFrame): 테스트 데이터 세트
        trained_model (LogisticRegression): 피팅된 모델

    Returns:
        forecast (pd.DataFrame): 예측 데이터 세트
    """
    print("     테스트 데이터 세트 예측 중...")
    X,y = test_dataset.iloc[:,:-1],test_dataset.iloc[:,-1]
    predictions = trained_model.predict_proba(X)[:, 1]
    print("     예측 완료!")
    return predictions

def roc_from_scratch(probabilities, test_dataset, partitions=100):
    print("     ROC 곡선의 계산...")
    y_test = test_dataset.iloc[:,-1]
    
    roc = np.array([])
    for i in range(partitions + 1):
        threshold_vector = np.greater_equal(probabilities, i / partitions).astype(int)
        tpr, fpr = true_false_positive(threshold_vector, y_test)
        roc = np.append(roc, [fpr, tpr])
    
    roc_np = roc.reshape(-1, 2)
    roc_data = pd.DataFrame({"False positive rate": roc_np[:, 0], "True positive rate": roc_np[:, 1]})
    print("     계산 완료")
    print("     스코어링중...")

    score_auc = roc_auc_score(y_test, probabilities)
    print("     스코어링 완료\n")

    return roc_data, score_auc


def true_false_positive(threshold_vector:np.array, y_test:np.array):
    """참양성률과 거짓양성률을 계산하는 함수
    
    Args:
        threshold_vector (np.array): 테스트 데이터 세트
        y_test (np.array): 피팅된 모델

    Returns:
        tpr (pd.DataFrame): 예측된 데이터 세트
        fpr (pd.DataFrame): 예측된 데이터 세트
    """
    
    true_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 1)
    true_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 0)
    false_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 0)
    false_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 1)

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

    return tpr, fpr

def create_metrics(predictions:np.array, test_dataset:np.array):
    print("     메트릭 생성 중...")
    threshold = 0.5
    threshold_vector = np.greater_equal(predictions, threshold).astype(int)
    
    y_test = test_dataset.iloc[:,-1]
    
    true_positive = (np.equal(threshold_vector, 1) & np.equal(y_test, 1)).sum()
    true_negative = (np.equal(threshold_vector, 0) & np.equal(y_test, 0)).sum()
    false_positive = (np.equal(threshold_vector, 1) & np.equal(y_test, 0)).sum()
    false_negative = (np.equal(threshold_vector, 0) & np.equal(y_test, 1)).sum()


    f1_score = np.around(2*true_positive/(2*true_positive+false_positive+false_negative), decimals=2)
    accuracy = np.around((true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative), decimals=2)
    dict_ftpn = {"tp": true_positive, "tn": true_negative, "fp": false_positive, "fn": false_negative}
    
    
    number_of_good_predictions = true_positive + true_negative
    number_of_false_predictions = false_positive + false_negative
    
    metrics = {"f1_score": f1_score,
               "accuracy": accuracy,
               "dict_ftpn": dict_ftpn,
               'number_of_predictions': len(predictions),
               'number_of_good_predictions':number_of_good_predictions,
               'number_of_false_predictions':number_of_false_predictions}
    
    print("     메트릭 생성 완료!")
    return metrics

    
def create_results(forecast_values,test_dataset):
    forecast_series_proba = pd.Series(np.around(forecast_values,decimals=2), index=test_dataset.index, name='Probability')
    forecast_series = pd.Series((forecast_values>0.5).astype(int), index=test_dataset.index, name='Forecast')
    true_series = pd.Series(test_dataset.iloc[:,-1], name="Historical",index=test_dataset.index)
    index_series = pd.Series(range(len(true_series)), index=test_dataset.index, name="Id")
    
    results = pd.concat([index_series, forecast_series_proba, forecast_series, true_series], axis=1)
    return results
