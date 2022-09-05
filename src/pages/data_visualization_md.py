import pandas as pd
import numpy as np


dv_graph_selector = ['Histogram','Scatter']
dv_graph_selected = dv_graph_selector[0]

# 히스토그램 대화 상자
dv_width_histo = "100%"
dv_height_histo = 600

dv_dict_overlay = {'barmode':'overlay', "margin":{"t":20}}

dv_select_x_ = ['CREDITSCORE', 'AGE', 'TENURE', 'BALANCE', 'NUMOFPRODUCTS', 'HASCRCARD', 'ISACTIVEMEMBER', 'ESTIMATEDSALARY', 'GEOGRAPHY_FRANCE', 'GEOGRAPHY_GERMANY', 'GEOGRAPHY_SPAIN', 'GENDER_MALE']


def creation_scatter_dataset(test_dataset:pd.DataFrame):
    """이 함수는 산점도에 대한 데이터셋을 생성합니다. 모든 열(Exited 제외)에 대해 양수 및 음수 버전이 있습니다.
    양수 열은 Exited가 0일 때 NaN을 가지며 음수 열은 Exited가 1일 때 NaN을 갖습니다.

    Args:
        test_dataset (pd.DataFrame): 테스트 데이터 세트
        
    Returns:
        pd.DataFrame: 데이터프레임
    """
    scatter_dataset = test_dataset.copy()

    for column in scatter_dataset.columns:
        if column != 'EXITED' :
            column_neg = str(column)+'_neg'
            column_pos = str(column)+'_pos'
            
            scatter_dataset[column_neg] = scatter_dataset[column]
            scatter_dataset[column_pos] = scatter_dataset[column]
            
            scatter_dataset.loc[(scatter_dataset['EXITED'] == 1),column_neg] = np.NaN
            scatter_dataset.loc[(scatter_dataset['EXITED'] == 0),column_pos] = np.NaN
    
    return scatter_dataset


def creation_of_dialog_scatter(column, state=None):
    """이 코드는 산점도에 사용되는 Markdown을 생성합니다. 부분(다시 로드할 수 있는 미니 페이지)을 변경하는 데 사용됩니다. 
    선택기가 생성되고 그래프의 x 및 y는 여기에서 변경하여 결정됩니다. 
    속성의 사전도 사용하는 열에 따라 변경됩니다.
    """
    if column == 'AGE' or column == 'CREDITSCORE' and state is not None:
        state.dv_dict_overlay = {'barmode':'overlay',"margin":{"t":20}}
    elif state is not None:
        state.dv_dict_overlay = {"margin":{"t":20}}
        
    md = """
<|layout|columns= 1 1 1|columns[mobile]=1|
<|
Type of graph \n \n <|{dv_graph_selected}|selector|lov={dv_graph_selector}|dropdown|>
|>

<|
Select **x** \n \n <|{x_selected}|selector|lov={select_x}|dropdown=True|>
|>

<|
Select **y** \n \n <|{y_selected}|selector|lov={select_y}|dropdown=True|>
|>
|>

<|part|render={x_selected=='"""+column+"""'}|
<|{scatter_dataset}|chart|x="""+column+"""|y[1]={y_selected+'_pos'}|y[2]={y_selected+'_neg'}|color[1]=red|color[2]=green|name[1]=Exited|name[2]=Stayed|height={dv_height_histo}|width={dv_width_histo}|mode=markers|type=scatter|layout={dv_dict_overlay}|>
|>
"""
    return md



def creation_histo_full(test_dataset:pd.DataFrame):
    """이 함수는 히스토그램 플롯에 대한 데이터 세트를 생성합니다. 모든 열(Exited 제외)에 대해 양수 및 음수 버전이 있습니다.
    Exited가 0일 때 양수 열에는 NaN이 있고 Exited가 1일 때 음수 열에는 NaN이 있습니다. 

    Args:
        test_dataset (pd.DataFrame): 테스트 데이터세트

    Returns:
        pd.DataFrame: 히스토그램을 표시하는 데 사용되는 데이터 프레임
    """
    histo_full = test_dataset.copy()
    # 각 클래스에 대해 동일한 수의 포인트를 갖도록 결정적 오버샘플링을 생성합니다.
    histo_1 = histo_full.loc[histo_full['EXITED'] == 1]    
    
    frames = [histo_full,histo_1,histo_1,histo_1]
    
    histo_full = pd.concat(frames, sort=False)
    
    for column in histo_full.columns:
        column_neg = str(column)+'_neg'
        histo_full[column_neg] = histo_full[column]
        histo_full.loc[(histo_full['EXITED'] == 1),column_neg] = np.NaN
        histo_full.loc[(histo_full['EXITED'] == 0),column] = np.NaN
        
    return histo_full


def creation_of_dialog_histogram(column, state=None):
    """이 코드는 히스토그램 플롯에 사용되는 마크다운을 생성합니다. 부분(다시 로드할 수 있는 미니 페이지)을 변경하는 데 사용됩니다. 
    선택기가 생성되고 그래프의 x는 여기에서 변경하여 결정됩니다. 
    속성의 사전도 사용하는 열에 따라 변경됩니다.
    """
    if column == 'AGE' or column == 'CREDITSCORE' and state is not None:
        state.dv_dict_overlay = {'barmode':'overlay',"margin":{"t":20}}
    elif state is not None:
        state.dv_dict_overlay = {"margin":{"t":20}}
        
    md = """
<|layout|columns= 1 1 1|columns[mobile]=1|
<|
Select type of graph : \n \n <|{dv_graph_selected}|selector|lov={dv_graph_selector}|dropdown|>
|>

<|
Select **x**: \n \n <|{x_selected}|selector|lov={select_x}|dropdown=True|>
|>
|>


<|{histo_full[['"""+column+"""','"""+column+"""_neg','EXITED']]}|chart|type=histogram|x[1]="""+column+"""|x[2]="""+column+"""_neg|y=EXITED|label=EXITED|color[1]=red|color[2]=green|name[1]=Exited|name[2]=Stayed|height={dv_height_histo}|width={dv_width_histo}|layout={dv_dict_overlay}|class_name=histogram|>
"""
    return md

dv_data_visualization_md = """
# 데이터 시각화

<|part|render={dv_graph_selected == 'Histogram'}|partial={partial_histo}|>

<|part|render={dv_graph_selected == 'Scatter'}|partial={partial_scatter}|>

"""

