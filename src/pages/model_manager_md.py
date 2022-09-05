import pandas as pd
import numpy as np


mm_select_x_ = ['CREDITSCORE', 'AGE', 'TENURE', 'BALANCE', 'NUMOFPRODUCTS', 'HASCRCARD', 'ISACTIVEMEMBER', 'ESTIMATEDSALARY', 'GEOGRAPHY_FRANCE', 'GEOGRAPHY_GERMANY', 'GEOGRAPHY_SPAIN', 'GENDER_MALE']

mm_graph_selector_scenario = ['Metrics', 'Features', 'Histogram','Scatter']
mm_graph_selected_scenario = mm_graph_selector_scenario[0]

mm_algorithm_selector = ['Baseline','ML']
mm_algorithm_selected = mm_algorithm_selector[0]

mm_pie_color_dict_2 ={"piecolorway":["#00D08A","#FE913C"]}
mm_pie_color_dict_4 = {"piecolorway":["#00D08A","#81F1A0","#F3C178","#FE913C"]}

mm_height_histo = 530


mm_margin_features = {'margin': {'l': 150, 'r': 50, 'b': 50, 't': 20}}

def creation_scatter_dataset_pred(test_dataset:pd.DataFrame, forecast_series:pd.Series):
    """이 함수는 예측을 위한 산점도에 대한 데이터 세트를 생성합니다. 모든 열(EXITED 제외)에 대해 양수 및 음수 버전이 있습니다.
    EXITED는 예측이 좋은지 나쁜지를 나타내는 바이너리입니다.
    양수 열은 Exited가 0일 때 NaN을 가지며 음수 열은 Exited가 1일 때 NaN을 갖습니다. 

    Args:
        test_dataset (pd.DataFrame): 테스트 데이터 세트
        forecast_series (pd.DataFrame): 예측 데이터 세트

    Returns:
        pd.DataFrame: 히스토그램을 표시하는 데 사용되는 데이터 프레임
    """
    
    scatter_dataset = test_dataset.copy()
    scatter_dataset['EXITED'] =  (scatter_dataset['EXITED']!=forecast_series.to_numpy()).astype(int)

    for column in scatter_dataset.columns:
        if column != 'EXITED' :
            column_neg = str(column)+'_neg'
            column_pos = str(column)+'_pos'
            
            scatter_dataset[column_neg] = scatter_dataset[column]
            scatter_dataset[column_pos] = scatter_dataset[column]
            
            scatter_dataset.loc[(scatter_dataset['EXITED'] == 1),column_neg] = np.NaN
            scatter_dataset.loc[(scatter_dataset['EXITED'] == 0),column_pos] = np.NaN
    
    return scatter_dataset


def creation_of_dialog_scatter_pred(column, state=None):
    """이 코드는 예측을 위한 산점도에 사용되는 마크다운을 생성합니다. 부분(다시 로드할 수 있는 미니 페이지)을 변경하는 데 사용됩니다. 
    선택기가 생성되고 그래프의 x 및 y는 여기에서 변경하여 결정됩니다. 
    속성의 사전도 사용하는 열에 따라 변경됩니다.
    """
    if column == 'AGE' or column == 'CREDITSCORE' and state is not None:
        state.dv_dict_overlay = {'barmode':'overlay',"margin":{"t":20}}
    elif state is not None:
        state.dv_dict_overlay = {"margin":{"t":20}}
        
    md = """
<|layout|columns= 1 1|columns[mobile]=1|
<|
Select **x** \n \n <|{x_selected}|selector|lov={select_x}|dropdown|>
|>

<|
Select **y** \n \n <|{y_selected}|selector|lov={select_y}|dropdown|>
|>
|>

<|{scatter_dataset_pred}|chart|x="""+column+"""|y[1]={y_selected+'_pos'}|y[2]={y_selected+'_neg'}|color[1]=red|color[2]=green|name[1]=Bad prediction|name[2]=Good prediction|height={mm_height_histo}|width={dv_width_histo}|mode=markers|type=scatter|layout={dv_dict_overlay}|>

"""
    return md


def creation_histo_full_pred(test_dataset:pd.DataFrame,forecast_series:pd.Series):
    """이 함수는 예측에 대한 히스토그램 플롯에 대한 데이터 세트를 생성합니다. 모든 열(PREDICTION 제외)에 대해 양수 및 음수 버전이 있습니다.
    PREDICTION은 예측이 좋은지 나쁜지를 나타내는 이진법입니다.
    PREDICTION이 0일 때 양수 열에는 NaN이 있고 PREDICTION이 1일 때 음수 열에는 NaN이 있습니다. 

    Args:
        test_dataset (pd.DataFrame): 테스트 데이터 세트
        Forecast_series (pd.DataFrame): 예측 데이터 세트

    Returns:
        pd.DataFrame: 히스토그램을 표시하는 데 사용되는 데이터 프레임
    """
    histo_full = test_dataset.copy()
    histo_full['EXITED'] =  (histo_full['EXITED']!=forecast_series.to_numpy()).astype(int)
    histo_full.columns = histo_full.columns.str.replace('EXITED', 'PREDICTION')
    
    for column in histo_full.columns:
        column_neg = str(column)+'_neg'
        histo_full[column_neg] = histo_full[column]
        histo_full.loc[(histo_full['PREDICTION'] == 1),column_neg] = np.NaN
        histo_full.loc[(histo_full['PREDICTION'] == 0),column] = np.NaN
        
    return histo_full


metrics_md = """
<br/>
<|layout|columns=1 1 1|columns[mobile]=1|

<|
<|{accuracy}|indicator|value={accuracy}|min=0|max=1|width=200px|>
<center>
**Model accuracy**
</center>
<|{pie_plotly}|chart|x=values|label=labels|title=Accuracy of predictions model|height={height_plotly}|width=100%|type=pie|layout={mm_pie_color_dict_2}|>
|>

<|
<|{score_auc}|indicator|value={score_auc}|min=0|max=1|width=200px|>
<center>
**Model AUC**
</center>
<|{pie_confusion_matrix}|chart|x=values|label=labels|title=Confusion Matrix|height={height_plotly}|width=100%|type=pie|layout={mm_pie_color_dict_4}|>
|>

<|
<|{f1_score}|indicator|value={f1_score}|min=0|max=1|width=200px|>
<center>
**Model F1-score**
</center>
<|{distrib_class}|chart|x=values|label=labels|title=Distribution between Exited and Stayed|height={height_plotly}|width=100%|type=pie|layout={mm_pie_color_dict_2}|>
|>

|>
"""


features_md = """
<|{features_table}|chart|type=bar|y=Features|x=Importance|orientation=h|layout={mm_margin_features}|>
"""

def creation_of_dialog_histogram_pred(column, state=None):
    """이 코드는 산점도에 사용된 마크다운을 생성합니다. 부분(다시 로드할 수 있는 미니 페이지)을 변경하는 데 사용됩니다. 
    선택자가 생성되고 여기에서 변경하여 그래프의 x가 결정됩니다. 
    속성의 사전도 사용하는 열에 따라 변경됩니다.
    """
    if column == 'AGE' or column == 'CREDITSCORE' and state is not None:
        state.dv_dict_overlay = {'barmode':'overlay',"margin":{"t":20}}
    elif state is not None:
        state.dv_dict_overlay = {"margin":{"t":20}}
        
    md = """
<|
Select **x** \n \n <|{x_selected}|selector|lov={select_x}|dropdown=True|>
|>

<|{histo_full_pred[['"""+column+"""','"""+column+"""_neg','PREDICTION']]}|chart|type=histogram|x[1]="""+column+"""|x[2]="""+column+"""_neg|y=PREDICTION|label=PREDICTION|color[1]=red|color[2]=green|name[1]=Bad prediction|name[2]=Good prediction|height={mm_height_histo}|width={dv_width_histo}|layout={dv_dict_overlay}|class_name=histogram|>

"""
    return md


mm_model_manager_md = """
# 모델 매니저

<|layout|columns=1 1 1 1|columns[mobile]=1|
Algorithm
<|{mm_algorithm_selected}|selector|lov={mm_algorithm_selector}|dropdown=True|>

Type of graph
<|{mm_graph_selected_scenario}|selector|lov={mm_graph_selector_scenario}|dropdown=True|>

<br/>
<br/>
<center>
<|show roc|button|on_action=show_roc_fct|>
</center>

<br/>
<br/>
<center>
**Number of predictions: ** *<|{number_of_predictions}|>*
</center>
|>

<|part|render={mm_graph_selected_scenario == 'Metrics'}|
"""+metrics_md+"""
|>

<|part|render={mm_graph_selected_scenario == 'Features'}|
"""+features_md+"""
|>

<|part|render={mm_graph_selected_scenario == 'Scatter'}|partial={partial_scatter_pred}|>

<|part|render={mm_graph_selected_scenario == 'Histogram'}|partial={partial_histo_pred}|>

"""
