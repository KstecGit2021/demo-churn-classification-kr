# 혼동 행렬 대화 상자
db_confusion_matrix_md = """
<|part|render={db_table_selected=='Confusion Matrix'}|
<center>
<|{score_table}|table|height=200px|width=400px|show_all=True|>
</center>
|>
"""

# 학습 데이터 세트에 대한 테이블
db_train_dataset_md = """
<|part|render={db_table_selected=='Training Dataset'}|
<|{train_dataset}|table|width=1400px|height=560px|>
|>
"""

# 예측 데이터 세트에 대한 테이블
db_forecast_dataset_md = """
<|part|render={db_table_selected=='Forecast Dataset'}|
<center>
<|{values}|table|height=560px|width=800px|style={get_style}|>
</center>
|>
"""

# 테스트 데이터 세트에 대한 테이블
db_test_dataset_md = """
<|part|render={db_table_selected=='Test Dataset'}|
<|{test_dataset}|table|width=1400px|height=560px|>
|>
"""

# 표시할 테이블을 선택하는 선택기
db_table_selector = ['Training Dataset', 'Test Dataset', 'Forecast Dataset', 'Confusion Matrix']
db_table_selected = db_table_selector[0]

# 전체 페이지를 만들기 위한 문자열 집계
db_databases_md = """
# 데이터베이스

<|layout|columns=2 2 1|columns[mobile]=1|
<|
**Algorithm**: \n \n <|{mm_algorithm_selected}|selector|lov={mm_algorithm_selector}|dropdown=True|>
|>

<|
**Table**: \n \n <|{db_table_selected}|selector|lov={db_table_selector}|dropdown=True|>
|>

<br/>
<br/>
<|{PATH_TO_TABLE}|file_download|name=table.csv|label=Download table|>
|>
""" + db_test_dataset_md + db_confusion_matrix_md + db_train_dataset_md + db_forecast_dataset_md


