import numpy as np
import pandas as pd

#--Path
train_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/train'
test_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/test'
model_data_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/model_data'


#---Train Dataframes
patient_df= pd.read_csv(model_data_path+'/demographics_train.csv')
observations_df= pd.read_csv(train_path+'/observations.csv')

observations_df=observations_df[observations_df.DATE.str[:4].astype(int)==2020]
df1=pd.merge(patient_df, observations_df, how='inner', left_on='Id', right_on='PATIENT')
df1=df1[(df1['covid_flag'] == 0) | (df1['covid_flag'] == 1)& pd.notnull(df1['covid_flag'])]

df1.DATE=pd.to_datetime(df1.DATE)
df1.Index_date=pd.to_datetime(df1.Index_date)
df2=df1[(df1.Index_date-df1.DATE).dt.days>=21]

df2=df2[['PATIENT', 'DESCRIPTION', 'covid_flag']]
df3=df2.drop_duplicates()
df_obs_pivot=df3.pivot_table(index=('PATIENT','covid_flag'),columns='DESCRIPTION', aggfunc='size',fill_value=0).reset_index();
df_obs_pivot= df_obs_pivot.drop(['SARS-CoV-2 RNA Pnl Resp NAA+probe','Adenovirus A+B+C+D+E DNA [Presence] in Respiratory specimen by NAA with probe detection'],axis=1)
df_obs_pivot.to_csv(model_data_path+'/observation_pivot_train.csv')

#---Test Dataframes
patient_test= pd.read_csv(model_data_path+'/demographics_test.csv')
observations_test= pd.read_csv(test_path+'/observations.csv')

df1=pd.merge(patient_test, observations_test, how='inner', left_on='Id', right_on='PATIENT')
df2=df1[['PATIENT', 'DESCRIPTION']]
df3=df2.drop_duplicates()
df_obs_pivot=df3.pivot_table(index=('PATIENT'),columns='DESCRIPTION', aggfunc='size',fill_value=0).reset_index();
df_obs_pivot.to_csv(model_data_path+'/observation_pivot_test.csv')
