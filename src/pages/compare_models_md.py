from sklearn.metrics import f1_score

import pandas as pd
import numpy as np

cm_height_histo = "100%"
cm_dict_barmode = {"barmode": "stack","margin":{"t":30}}
cm_options_md = "height={cm_height_histo}|width={cm_height_histo}|layout={cm_dict_barmode}"

cm_compare_models_md = """
# 모델 비교

<br/>
<br/>
<br/>

<|layout|columns= 1 1 1|columns[mobile]=1|
<|{accuracy_graph}|chart|type=bar|x=Pipeline|y[1]=Accuracy Model|y[2]=Accuracy Baseline|title=Accuracy|""" + cm_options_md + """|>

<|{f1_score_graph}|chart|type=bar|x=Pipeline|y[1]=F1 Score Model|y[2]=F1 Score Baseline|title=F1 Score|""" + cm_options_md + """|>

<|{score_auc_graph}|chart|type=bar|x=Pipeline|y[1]=AUC Score Model|y[2]=AUC Score Baseline|title=AUC Score|""" + cm_options_md + """|>

|>
"""

def c_update_metrics(scenario, pipeline):
    """이 함수는 파이프라인을 사용하여 시나리오의 메트릭을 업데이트합니다.

    Args:
        scenario (scenario): 선택한 시나리오
        pipeline (str): 선택한 파이프라인의 이름

    Returns:
        obj: 여러 값, 메트릭을 나타내는 목록
    """
    metrics = scenario.pipelines[pipeline].metrics.read()

    number_of_predictions = metrics['number_of_predictions']
    number_of_good_predictions = metrics['number_of_good_predictions']
    number_of_false_predictions = metrics['number_of_false_predictions']

    accuracy = np.around(metrics['accuracy'], decimals=2)
    f1_score = np.around(metrics['f1_score'], decimals=2)
    score_auc = np.around(scenario.pipelines[pipeline].score_auc.read(),decimals=2)
    
    dict_ftpn = metrics['dict_ftpn']
    
    fp_ = dict_ftpn['fp']
    tp_ = dict_ftpn['tp']
    fn_ = dict_ftpn['fn']
    tn_ = dict_ftpn['tn']
    
    return number_of_predictions, accuracy, f1_score, score_auc, number_of_good_predictions, number_of_false_predictions, fp_, tp_, fn_, tn_


def compare_charts(accuracies, f1_scores, scores_auc, names):
    """이 함수는 모델 비교 페이지에서 사용되는 pandas 데이터 프레임(차트)을 생성합니다.

    Args:
        accuracies (list): 정확도 목록
        f1_scores (list): f1 점수 목록
        scores_auc (list): auc 점수 목록
        names (list): 시나리오 이름 목록

    Returns:
        pd.DataFrame: 세 개의 데이터프레임의 결과
    """
    accuracy_graph = pd.DataFrame(create_metric_dict(accuracies, "Accuracy", names))
    f1_score_graph = pd.DataFrame(create_metric_dict(f1_scores, "F1 Score", names))
    score_auc_graph = pd.DataFrame(create_metric_dict(scores_auc, "AUC Score", names))

    return accuracy_graph, f1_score_graph, score_auc_graph

def compare_models_baseline(scenario,pipelines):
    """이 함수는 파이프라인 비교를 위한 개체를 생성합니다.

    Args:
        scenario (scenario): 선택한 시나리오
        pipelines (str): 선택한 파이프라인의 이름

    Returns:
        pd.DataFrame: 세 개의 데이터프레임의 결과
    """
    accuracies = []
    f1_scores = []
    scores_auc = []
    names = []
    for pipeline in pipelines:
        (_,accuracy,f1_score,score_auc,_,_,_,_,_,_) = c_update_metrics(scenario, pipeline)
        
        accuracies.append(accuracy)
        f1_scores.append(f1_score)
        scores_auc.append(score_auc)
        names.append(pipeline[9:])
        
    accuracy_graph,f1_score_graph, score_auc_graph = compare_charts(accuracies, f1_scores, scores_auc, names)
    return accuracy_graph, f1_score_graph, score_auc_graph
    

def create_metric_dict(metric, metric_name, names):
    """이 함수는 Gui에 표시된 데이터 프레임에서 사용될 여러 파이프라인에 대한 메트릭 사전을 생성합니다.

    Args:
        metric (list): 메트릭 값
        metric_name (str): 메트릭의 이름
        names (list): 시나리오 이름 목록

    Returns:
        dict: pandas 데이터프래임에 사용되는 사전
    """
    metric_dict = {}
    initial_list = [0]*len(names)
    
    metric_dict["Pipeline"] = names
    for i in range(len(names)):
        current_list = initial_list.copy()
        
        current_list[i] = metric[i]
        metric_dict[metric_name +" "+ names[i].capitalize()] = current_list
        
    return metric_dict


