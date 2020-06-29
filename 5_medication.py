import numpy as np
import pandas as pd

#--Path
train_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/train'
test_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/test'
model_data_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/model_data'


#---Train Dataframes
patient_df= pd.read_csv(model_data_path+'/demographics_train.csv')
medications_df= pd.read_csv(train_path+'/medications.csv')
df1=pd.merge(patient_df, medications_df, how='inner', left_on='Id', right_on='PATIENT')
df1=df1[(df1['covid_flag'] == 0) | (df1['covid_flag'] == 1)& pd.notnull(df1['covid_flag'])]
df1.START=pd.to_datetime(df1.START)
df1.STOP=pd.to_datetime(df1.STOP)
df1.Index_date=pd.to_datetime(df1.Index_date)
df2=df1[(df1.Index_date-df1.START).dt.days>=21 & (df1.STOP>df1.Index_date)  | (df1.STOP == 'NaT')]
df2=df2[['PATIENT', 'DESCRIPTION', 'covid_flag']]
df3=df2.drop_duplicates()
df_med_pivot=df3.pivot_table(index=('PATIENT','covid_flag'),columns='DESCRIPTION', aggfunc='size',fill_value=0).reset_index();
df_med_pivot= df_med_pivot.drop(['Acetaminophen 500 MG Oral Tablet','0.4 ML Enoxaparin sodium 100 MG/ML Prefilled Syringe'],axis=1)
df_med_pivot.to_csv(model_data_path+'/medication_pivot_train.csv')

#---Test Dataframes
patient_test= pd.read_csv(model_data_path+'/demographics_test.csv')
medications_test= pd.read_csv(test_path+'/medications.csv')
df1=pd.merge(patient_test, medications_test, how='inner', left_on='Id', right_on='PATIENT')
df2=df1[['PATIENT', 'DESCRIPTION']]
df3=df2.drop_duplicates()
df_med_pivot=df3.pivot_table(index=('PATIENT'),columns='DESCRIPTION', aggfunc='size',fill_value=0).reset_index();
df_med_pivot.to_csv(model_data_path+'/medication_pivot_test.csv')
