import numpy as np
import pandas as pd

#--Path
train_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/train'
test_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/test'
model_data_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/model_data'


#---Train Dataframes
patient_df= pd.read_csv(model_data_path+'/demographics_train.csv')
procedure_df= pd.read_csv(train_path+'/procedures.csv')

df1=pd.merge(patient_df, procedure_df, how='inner', left_on='Id', right_on='PATIENT')
df1=df1[(df1['covid_flag'] == 0) | (df1['covid_flag'] == 1)]
df1.DATE=pd.to_datetime(df1.DATE)
df1.Index_date=pd.to_datetime(df1.Index_date)
df2=df1[(df1.Index_date-df1.DATE).dt.days>=21]

df2=df1[['PATIENT', 'DESCRIPTION', 'covid_flag']]
df3=df2.drop_duplicates()
df_proc_pivot=df3.pivot_table(index=('PATIENT','covid_flag'),columns='DESCRIPTION', aggfunc='size',fill_value=0).reset_index();
df_proc_pivot= df_proc_pivot.drop(['Oxygen administration by mask (procedure)','Face mask (physical object)','Placing subject in prone position (procedure)'],axis=1)
df_proc_pivot.to_csv(model_data_path+'/procedure_pivot_train.csv')

#---Test Dataframes
patient_test= pd.read_csv(model_data_path+'/demographics_test.csv')
procedure_test= pd.read_csv(test_path+'/procedures.csv')

df1=pd.merge(patient_test, procedure_test, how='inner', left_on='Id', right_on='PATIENT')
df2=df1[['PATIENT', 'DESCRIPTION']]
df3=df2.drop_duplicates()
df_proc_test=df3.pivot_table(index=('PATIENT'),columns='DESCRIPTION', aggfunc='size',fill_value=0).reset_index();
df_proc_test.to_csv(model_data_path+'/procedure_pivot_test.csv')

