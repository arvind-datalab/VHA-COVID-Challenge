import numpy as np
import pandas as pd

#--Path
train_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/train'
test_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/test'
model_data_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/model_data'


#---Dataframes
patient_df= pd.read_csv(model_data_path+'/demographics_train.csv')
condition_df= pd.read_csv(train_path+'/conditions.csv')
conditiontype_df= pd.read_csv(model_data_path+'/Condition_Type.csv')

df1=pd.merge(patient_df, condition_df, how='inner', left_on='Id', right_on='PATIENT')
df1=pd.merge(df1, conditiontype_df, how='left', left_on='DESCRIPTION', right_on='DESCRIPTION')

df1=df1[((df1['covid_flag'] == 0) | (df1['covid_flag'] == 1)) & (df1['Type'] == 'Finding')]
df1.START=pd.to_datetime(df1.START)
df1.Index_date=pd.to_datetime(df1.Index_date)
df2=df1[((df1.Index_date-df1.START).dt.days<=14) & ((df1.Index_date-df1.START).dt.days>=0)]

df2=df2[['PATIENT', 'DESCRIPTION', 'covid_flag']]
df3=df2.drop_duplicates()
df_find_pivot=df3.pivot_table(index=('PATIENT','covid_flag'),columns='DESCRIPTION', aggfunc='size',fill_value=0).reset_index();
df_find_pivot.to_csv(model_data_path+'/Finding.csv')

d=condition_df[condition_df['PATIENT']=='0000641f-540d-408b-ad16-93c2de94f446']