# 기본 패키지
import pandas as pd

# 타이피 함수
import taipy as tp
from taipy.gui import Gui, Icon

# 구성 가져오기
from config.config import scenario_cfg
from taipy.core.config.config import Config

import os

# 임시 파일을 생성하기 위해 import
import pathlib

# 이 경로는 Datasouces 페이지에서 테이블을 다운로드할 수 있는 임시 파일을 만드는 데 사용됩니다.
# 
tempdir = pathlib.Path(".tmp")
tempdir.mkdir(exist_ok=True)
PATH_TO_TABLE = str(tempdir / "table.csv")

###############################################################################
# 데이터 저장소를 청소합니다.
###############################################################################

Config.configure_global_app(clean_entities_enabled=True)
tp.clean_all_entities()

##############################################################################################################################
# 시나리오 실행
##############################################################################################################################

def create_first_scenario(scenario_cfg):
    global scenario
    scenario = tp.create_scenario(scenario_cfg)
    tp.submit(scenario)
    
create_first_scenario(scenario_cfg)

# ############################################################################################################################
# 초기화 - 시나리오의 값을 읽을 수 있습니다.
##############################################################################################################################
forecast_values_baseline = scenario.pipelines['pipeline_baseline'].forecast_dataset.read()
forecast_values = scenario.pipelines['pipeline_model'].forecast_dataset.read()

test_dataset = scenario.pipelines['pipeline_baseline'].test_dataset.read()
train_dataset = scenario.pipelines['pipeline_preprocessing'].train_dataset.read()
roc_dataset = scenario.pipelines['pipeline_baseline'].roc_data.read()

test_dataset.columns = [str(column).upper() for column in test_dataset.columns]

# 히스토그램과 산점도를 이용한 데이터 시각화를 위한 것입니다.
select_x = test_dataset.drop('EXITED',axis=1).columns.tolist()
x_selected = select_x[0]

select_y = select_x
y_selected = select_y[1]

##############################################################################################################################
# 초기화 - 차트에 사용될 결과를 재개하는 데이터셋 생성
##############################################################################################################################
from pages.main_dialog import *

values_baseline = scenario.pipelines['pipeline_baseline'].results.read()
values_model = scenario.pipelines['pipeline_model'].results.read()

values = values_baseline.copy()

forecast_series = values['Forecast']
true_series = values['Historical']


scatter_dataset_pred = creation_scatter_dataset_pred(test_dataset,forecast_series)
histo_full_pred = creation_histo_full_pred(test_dataset,forecast_series)

histo_full = creation_histo_full(test_dataset)
scatter_dataset = creation_scatter_dataset(test_dataset)

features_table = scenario.pipelines['pipeline_train_baseline'].feature_importance.read()

# 올바른 파이프라인을 가져오는 일반 코드의 파이프라인 비교
# 
pipelines_to_compare = [pipeline for pipeline in scenario.pipelines if 'train' not in pipeline and 'preprocessing' not in pipeline]

accuracy_graph, f1_score_graph, score_auc_graph = compare_models_baseline(scenario, pipelines_to_compare) # comes from the compare_models.py

##############################################################################################################################
# 초기화 - 표시될 모델의 정확도와 클래스 분포를 보기 위한 파이 차트 생성
##############################################################################################################################
# '기준' 모델에 대한 메트릭을 계산합니다.
(number_of_predictions,
 accuracy, f1_score, score_auc,
 number_of_good_predictions,
 number_of_false_predictions,
 fp_, tp_, fn_, tn_) = c_update_metrics(scenario, 'pipeline_baseline')

# 파이 차트
pie_plotly = pd.DataFrame({"values": [number_of_good_predictions, number_of_false_predictions],
                           "labels": ["Correct predictions", "False predictions"]})

distrib_class = pd.DataFrame({"values": [len(values[values["Historical"]==0]),len(values[values["Historical"]==1])],
                              "labels" : ["Stayed", "Exited"]})

##############################################################################################################################
# 초기화 - 표시될 False/positive/negative/true 테이블 생성
##############################################################################################################################

score_table = pd.DataFrame({"Score":["Predicted stayed", "Predicted exited"],
                            "Stayed": [tn_, fp_],
                            "Exited" : [fn_, tp_]})

pie_confusion_matrix = pd.DataFrame({"values": [tp_,tn_,fp_,fn_],
                              "labels" : ["True Positive","True Negative","False Positive",  "False Negative"]})

##############################################################################################################################
# 초기화 - 그래픽 사용자 인터페이스 생성(상태)
##############################################################################################################################

# The list of pages that will be shown in the menu at the left of the page
menu_lov = [("Data Visualization", Icon('images/histogram_menu.svg', 'Data Visualization')),
            ("Model Manager", Icon('images/model.svg', 'Model Manager')),
            ("Compare Models", Icon('images/compare.svg', 'Compare Models')),
            ('Databases', Icon('images/Datanode.svg', 'Databases'))]

width_plotly = "450px"
height_plotly = "450px"

page_markdown = """
<|toggle|theme|>
<|menu|label=Menu|lov={menu_lov}|on_action=menu_fct|>

<|part|render={page == 'Data Visualization'}|
""" + dv_data_visualization_md + """
|>

<|part|render={page == 'Model Manager'}|
""" + mm_model_manager_md + """
|>

<|part|render={page == 'Compare Models'}|
""" + cm_compare_models_md + """
|>

<|part|render={page == 'Databases'}|
""" + db_databases_md + """
|>

"""

# 초기 페이지는 "시나리오 관리자" 페이지입니다.
page = "Data Visualization"
def menu_fct(state,var_name:str,fct,var_value):
    """메뉴 컨트롤에 변경이 있을 때 호출되는 함수

    Args:
        state : Taipy의 상태 객체
        var_name (str): 변경된 변수명
        var_value (obj): 변경된 변수 값
    """
    # 올바른 페이지를 렌더링하기 위해 state.page 변수의 값을 변경합니다.
    try :
        state.page = var_value['args'][0]
    except:
        print("Warning : No args were found")
    pass


# 예측 테이블을 위한 함수. 나쁜 예측은 빨간색이고 좋은 예측은 녹색입니다(css 클래스).
def get_style(state, index, row):
    return 'red' if row['Historical']!=row['Forecast'] else 'green'


##############################################################################################################################
# 전체 마크다운 생성
##############################################################################################################################


# dialog_md는 main_dialog.py에 있습니다.
# the other are found in the dialogs folder
entire_markdown = page_markdown + dialog_md

# 페이지 생성에 사용될 객체
gui = Gui(page=entire_markdown, css_file='main')
dialog_partial_roc = gui.add_partial(dialog_roc)

partial_scatter = gui.add_partial(creation_of_dialog_scatter(x_selected))
partial_histo = gui.add_partial(creation_of_dialog_histogram(x_selected))

partial_scatter_pred = gui.add_partial(creation_of_dialog_scatter_pred(x_selected))
partial_histo_pred = gui.add_partial(creation_of_dialog_histogram_pred(x_selected))

def update_partial_charts(state):
    """이 함수는 차트와 선택기를 포함하는 4개의 부분을 업데이트합니다. 
    Partials는 아래 기능을 사용하여 런타임에 다시 로드할 수 있는 미니 페이지입니다. 
    차트의 내용을 변경하기 위해 다시 로드됩니다.

    Args:
        state: GUI에서 사용되는 모든 변수를 포함하는 객체
    """
    state.partial_scatter.update_content(state, creation_of_dialog_scatter(state.x_selected, state))
    state.partial_histo.update_content(state, creation_of_dialog_histogram(state.x_selected, state))
    
    state.partial_scatter_pred.update_content(state, creation_of_dialog_scatter_pred(state.x_selected, state))
    state.partial_histo_pred.update_content(state, creation_of_dialog_histogram_pred(state.x_selected, state))


##############################################################################################################################
# 표시된 변수 업데이트
##############################################################################################################################


def update_variables(state, pipeline):
    """이 함수는 응용 프로그램에서 사용되는 다양한 변수와 데이터 프레임을 업데이트합니다.

    Args:
        state: GUI에서 사용되는 모든 변수를 포함하는 객체
        pipeline (str): 변수를 업데이트하는 데 사용되는 파이프라인의 이름
    """
    global scenario
    pipeline_str = 'pipeline_'+pipeline
    
    if pipeline == 'baseline':
        state.values = scenario.pipelines[pipeline_str].results.read()
    else:
        state.values = scenario.pipelines[pipeline_str].results.read()
        
    state.forecast_series = state.values['Forecast']
    state.true_series = state.values["Historical"]
    
    
    (state.number_of_predictions,
    state.accuracy, state.f1_score, state.score_auc,
    number_of_good_predictions, number_of_false_predictions,
    fp_, tp_, fn_, tn_) = c_update_metrics(scenario, pipeline_str)
    
    
    update_charts(state, pipeline_str, number_of_good_predictions, number_of_false_predictions, fp_, tp_, fn_, tn_)
    


def update_charts(state, pipeline_str, number_of_good_predictions, number_of_false_predictions, fp_, tp_, fn_, tn_):
    """이 함수는 GUI의 모든 차트를 업데이트합니다.

    Args:
        state: GUI에서 사용되는 모든 변수를 포함하는 객체
        pipeline_str(str): 표시된 파이프라인의 이름
        number_of_good_predictions(int): 좋은 예측의 수
        number_of_false_predictions(int): 잘못된 예측의 수
        fp_ (float): 위양성 비율
        tp_ (float): 참 긍정 비율
        fn_ (float): 위음성 비율
        tn_ (float): 참 음수 비율
    """
    state.roc_dataset = scenario.pipelines[pipeline_str].roc_data.read()
    
    if 'model' in pipeline_str:
        state.features_table = scenario.pipelines['pipeline_train_model'].feature_importance.read()
    elif 'baseline' in pipeline_str:
        state.features_table = scenario.pipelines['pipeline_train_baseline'].feature_importance.read()

    state.score_table = pd.DataFrame({"Score":["Predicted stayed", "Predicted exited"],
                                      "Stayed": [tn_, fp_],
                                      "Exited" : [fn_, tp_]})

    state.pie_confusion_matrix = pd.DataFrame({"values": [tp_, tn_, fp_, fn_],
                                               "labels" : ["True Positive", "True Negative", "False Positive", "False Negative"]})

    state.scatter_dataset_pred = creation_scatter_dataset_pred(test_dataset, state.forecast_series)
    state.histo_full_pred = creation_histo_full_pred(test_dataset, state.forecast_series)

    
    # 파이 차트
    state.pie_plotly = pd.DataFrame({"values": [number_of_good_predictions, number_of_false_predictions],
                                     "labels": ["Correct predictions", "False predictions"]})

    state.distrib_class = pd.DataFrame({"values": [len(state.values[state.values["Historical"]==0]),
                                                   len(state.values[state.values["Historical"]==1])],
                                        "labels" : ["Stayed", "Exited"]})


##############################################################################################################################
# on_change 함수
##############################################################################################################################

# 다른 기능은 frontend/dialogs의 오른쪽 폴더에 있습니다.
def on_change(state, var_name, var_value):
    """이 함수는 GUI에서 변수가 변경될 때 호출됩니다.

    Args:
        state : GUI에서 사용되는 모든 변수를 포함하는 객체
        var_name (str): 변경된 변수의 이름
        var_value (obj): 변경된 변수의 값
    """
    if var_name == 'x_selected' or var_name == 'y_selected':
        update_partial_charts(state)
    
    if var_name == 'mm_algorithm_selected':
        if var_value == 'Baseline':
            update_variables(state,'baseline')
        if var_value == 'ML':
            update_variables(state,'model')
        
    if (var_name == 'mm_algorithm_selected' or var_name == "db_table_selected" and state.page == 'Databases') or (var_name == 'page' and var_value == 'Databases'):
        # '데이터베이스' 페이지에 있는 경우 임시 csv 파일을 만들어야 합니다.
        handle_temp_csv_path(state)
           
    if var_name == 'page' and var_value != 'Databases':
        delete_temp_csv()



def delete_temp_csv():
    """이 함수는 임시 csv 파일을 삭제합니다."""
    if os.path.exists(PATH_TO_TABLE):
        os.remove(PATH_TO_TABLE)

def handle_temp_csv_path(state):
    """임시 csv 파일이 존재하는지 확인하는 함수입니다. 존재하면 삭제합니다. 
    그러면 임시 csv 파일이 오른쪽 테이블에 대해 생성됩니다.

    Args:
        state: GUI에서 사용되는 모든 변수를 포함하는 객체
    """
    if os.path.exists(PATH_TO_TABLE):
        os.remove(PATH_TO_TABLE)
    if state.db_table_selected == 'Test Dataset':
        state.test_dataset.to_csv(PATH_TO_TABLE, sep=';')
    if state.db_table_selected == 'Confusion Matrix':
        state.score_table.to_csv(PATH_TO_TABLE, sep=';')
    if state.db_table_selected == "Training Dataset":
        train_dataset.to_csv(PATH_TO_TABLE, sep=';')
    if state.db_table_selected == "Forecast Dataset":
        values.to_csv(PATH_TO_TABLE, sep=';')
    

##############################################################################################################################
# GUI 실행
##############################################################################################################################
if __name__ == '__main__':
    gui.run(title="Churn classification",
    		host='0.0.0.0',
    		port=os.environ.get('PORT', '5050'),
    		dark_mode=False)
else:
    app = gui.run(title="Churn classification",
    	         dark_mode=False,
                 run_server=False)
