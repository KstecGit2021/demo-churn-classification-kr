from algos.algos import *
from taipy import Scope, Frequency, Config
##############################################################################################################################
# 데이터노드 생성
##############################################################################################################################
# 데이터베이스에 연결하는 방법
path_to_csv = 'data/churn.csv'

# csv의 경로와 피클의 file_path
initial_dataset = Config.configure_data_node(id="initial_dataset",
                                             path=path_to_csv,
                                             storage_type="csv",
                                             has_header=True)

date_cfg = Config.configure_data_node(id="date", default_data="None")

preprocessed_dataset = Config.configure_data_node(id="preprocessed_dataset",
                                                  cacheable=True,
                                                  validity_period=dt.timedelta(days=1))

# 처리된 데이터를 포함하는 최종 데이터 노드
train_dataset = Config.configure_data_node(id="train_dataset",
                                           cacheable=True,
                                           validity_period=dt.timedelta(days=1))

# 처리된 데이터를 포함하는 최종 데이터 노드
trained_model = Config.configure_data_node(id="trained_model",
                                           cacheable=True,
                                           validity_period=dt.timedelta(days=1))

trained_model_baseline = Config.configure_data_node(id="trained_model_baseline",
                                                    cacheable=True,
                                                    validity_period=dt.timedelta(days=1))


# 처리된 데이터를 포함하는 최종 데이터 노드
test_dataset = Config.configure_data_node(id="test_dataset",
                                          cacheable=True,
                                          validity_period=dt.timedelta(days=1))

forecast_dataset = Config.configure_data_node(id="forecast_dataset",
                                          scope=Scope.PIPELINE,
                                             cacheable=True,
                                             validity_period=dt.timedelta(days=1))

roc_data = Config.configure_data_node(id="roc_data",
                                      scope=Scope.PIPELINE,
                                      cacheable=True,
                                      validity_period=dt.timedelta(days=1))

score_auc = Config.configure_data_node(id="score_auc",
                                       scope=Scope.PIPELINE,
                                       cacheable=True,
                                       validity_period=dt.timedelta(days=1))

metrics = Config.configure_data_node(id="metrics",
                                     scope=Scope.PIPELINE,
                                     cacheable=True,
                                     validity_period=dt.timedelta(days=1))

feature_importance_cfg = Config.configure_data_node(id="feature_importance",
                                                    scope=Scope.PIPELINE)


results = Config.configure_data_node(id="results",
                                     scope=Scope.PIPELINE,
                                     cacheable=True,
                                     validity_period=dt.timedelta(days=1))


##############################################################################################################################
# 작업 생성
##############################################################################################################################

# 작업은 함수를 실행하는 동안 입력 데이터 노드와 출력 데이터 노드 사이의 링크를 만듭니다. 
# 

# 초기 데이터세트 --> 데이터세트 전처리 --> 처리된 데이터세트
task_preprocess_dataset = Config.configure_task(id="preprocess_dataset",
                                                input=[initial_dataset,date_cfg],
                                                function=preprocess_dataset,
                                                output=preprocessed_dataset)

# 처리된 데이터세트 --> 학습 데이터 생성 --> 학습데이터세트, 테스트데이터세트
task_create_train_test = Config.configure_task(id="create_train_and_test_data",
                                               input=preprocessed_dataset,
                                               function=create_train_test_data,
                                               output=[train_dataset, test_dataset])


# 학습 데이터세트 --> 학습 모델 데이터 생성 --> 학습된 모델
task_train_model = Config.configure_task(id="train_model",
                                         input=train_dataset,
                                         function=train_model,
                                         output=[trained_model,feature_importance_cfg])
                                   
# 학습 데이터세트 --> 학습 모델 데이터 생성 --> 학습된 모델
task_train_model_baseline = Config.configure_task(id="train_model_baseline",
                                                  input=train_dataset,
                                                  function=train_model_baseline,
                                                  output=[trained_model_baseline,feature_importance_cfg])

# 테스트 데이터세트 --> 예측 --> 예측 데이터 세트
task_forecast = Config.configure_task(id="predict_the_test_data",
                                      input=[test_dataset, trained_model],
                                      function=forecast,
                                      output=forecast_dataset)

# 테스트 데이터세트 --> 예측 --> 예측 데이터 세트
task_forecast_baseline = Config.configure_task(id="predict_of_baseline",
                           input=[test_dataset, trained_model_baseline],
                           function=forecast_baseline,
                           output=forecast_dataset)

task_roc = Config.configure_task(id="task_roc",
                           input=[forecast_dataset, test_dataset],
                           function=roc_from_scratch,
                           output=[roc_data,score_auc])

task_roc_baseline = Config.configure_task(id="task_roc_baseline",
                           input=[forecast_dataset, test_dataset],
                           function=roc_from_scratch,
                           output=[roc_data,score_auc])

task_create_metrics = Config.configure_task(id="task_create_metrics",
                                            input=[forecast_dataset,test_dataset],
                                            function=create_metrics,
                                            output=metrics)

task_create_results = Config.configure_task(id="task_create_results",
                                            input=[forecast_dataset,test_dataset],
                                            function=create_results,
                                            output=results)



##############################################################################################################################
# 파이프라인 및 시나리오 생성
##############################################################################################################################

# 파이프라인 및 시나리오 구성
pipeline_preprocessing = Config.configure_pipeline(id="pipeline_preprocessing", task_configs=[task_preprocess_dataset,
                                                                                              task_create_train_test])

pipeline_train_baseline = Config.configure_pipeline(id="pipeline_train_baseline", task_configs=[task_train_model_baseline])
pipeline_train_model = Config.configure_pipeline(id="pipeline_train_model", task_configs=[task_train_model])

pipeline_model = Config.configure_pipeline(id="pipeline_model", task_configs=[task_forecast,
                                                                              task_roc,
                                                                              task_create_metrics,
                                                                              task_create_results])

pipeline_baseline = Config.configure_pipeline(id="pipeline_baseline", task_configs=[task_forecast_baseline,
                                                                                    task_roc_baseline,
                                                                                    task_create_metrics,
                                                                                    task_create_results])

# 시나리오는 파이프라인을 실행합니다.
scenario_cfg = Config.configure_scenario(id="churn_classification",
                                         pipeline_configs=[pipeline_preprocessing, 
                                                           pipeline_train_baseline, pipeline_train_model,
                                                           pipeline_model,pipeline_baseline],
                                         frequency=Frequency.WEEKLY)


