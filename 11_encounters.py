import numpy as np
import pandas as pd
import datetime
#--Path
train_path='C:/Users/anup.b.sharma/Accenture/Arvind, Chauhan - VA Challenge/Dataset/train'
test_path='C:/Users/anup.b.sharma/Accenture/Arvind, Chauhan - VA Challenge/Dataset/test'
model_data_path='C:/Users/anup.b.sharma/Accenture/Arvind, Chauhan - VA Challenge/Dataset/model_data'
#---Train Dataframes
patient_df= pd.read_csv(model_data_path+'/demographics_train.csv')
encounters_df= pd.read_csv(train_path+'/encounters.csv')
encounters_df1 = encounters_df[['PATIENT','START', 'ENCOUNTERCLASS']]
df1=pd.merge(patient_df, encounters_df1, how='inner', left_on='Id', right_on='PATIENT')
df1=df1[(df1['covid_flag'] == 0) | (df1['covid_flag'] == 1)& pd.notnull(df1['covid_flag'])]
df1.Index_date=pd.to_datetime(df1.Index_date)
df1.START = pd.to_datetime(df1.START).dt.strftime('%d/%m/%Y')
df1.START = pd.to_datetime(df1.START)
df2=df1[(df1.Index_date-df1.START).dt.days>=21]
df2=df2[['PATIENT', 'ENCOUNTERCLASS', 'covid_flag']]
encounters_df1_pivot=df2.pivot_table(index=('PATIENT','covid_flag'),columns='ENCOUNTERCLASS', aggfunc=len,fill_value=0).reset_index();
columns = encounters_df1_pivot.columns.tolist()
n =2
columns = columns[n:]
dfx = encounters_df1_pivot
def update(x, med):
    if x > med:
        return(1)
    else:
        return(0)

for x in columns:
    median_x = dfx[x].median() 
    dfx[x + '_med'] = dfx[x].apply(lambda x : update(x, median_x)) 
encounters_df2 = encounters_df[['PATIENT','START', 'REASONDESCRIPTION']]
df3=pd.merge(patient_df, encounters_df2, how='inner', left_on='Id', right_on='PATIENT')
df3=df3[(df3['covid_flag'] == 0) | (df3['covid_flag'] == 1)& pd.notnull(df3['covid_flag'])]
df3.Index_date=pd.to_datetime(df3.Index_date)
df3.START = pd.to_datetime(df3.START).dt.strftime('%d/%m/%Y')
df3.START = pd.to_datetime(df3.START)
df4=df3[(df3.Index_date-df3.START).dt.days>=21]
df4=df4[['PATIENT', 'REASONDESCRIPTION']]
encounters_df2_pivot=df4.pivot_table(index=('PATIENT'),columns='REASONDESCRIPTION', aggfunc='size',fill_value=0).reset_index();
dfy = encounters_df2_pivot
new_names = [(i,i+'_enc') for i in dfy.iloc[:, 2:].columns.values]
dfy.rename(columns = dict(new_names), inplace=True)
dfz=pd.merge(dfx, dfy, how='inner', left_on = 'PATIENT', right_on='PATIENT')
dfz.to_csv(model_data_path+'/encounter_pivot_train.csv')


#---Test Dataframes
patient_test= pd.read_csv(model_data_path+'/demographics_test.csv')
encounters_test= pd.read_csv(test_path+'/encounters.csv')
encounters_df1 = encounters_test[['PATIENT','START', 'ENCOUNTERCLASS']]
df1=pd.merge(patient_test, encounters_df1, how='inner', left_on='Id', right_on='PATIENT')
df2=df1[['PATIENT', 'ENCOUNTERCLASS']]
encounters_df1_pivot=df2.pivot_table(index=('PATIENT'),columns='ENCOUNTERCLASS', aggfunc=len,fill_value=0).reset_index();
columns = encounters_df1_pivot.columns.tolist()
n =2
columns = columns[n:]
dfx = encounters_df1_pivot
def update(x, med):
    if x > med:
        return(1)
    else:
        return(0)

for x in columns:
    median_x = dfx[x].median() 
    dfx[x + '_med'] = dfx[x].apply(lambda x : update(x, median_x)) 
encounters_df2 = encounters_test[['PATIENT','START', 'REASONDESCRIPTION']]
df3=pd.merge(patient_df, encounters_df2, how='inner', left_on='Id', right_on='PATIENT')
df4=df3[['PATIENT', 'REASONDESCRIPTION']]
encounters_df2_pivot=df4.pivot_table(index=('PATIENT'),columns='REASONDESCRIPTION', aggfunc='size',fill_value=0).reset_index();
dfy = encounters_df2_pivot
new_names = [(i,i+'_enc') for i in dfy.iloc[:, 2:].columns.values]
dfy.rename(columns = dict(new_names), inplace=True)
dfz=pd.merge(dfx, dfy, how='inner', left_on = 'PATIENT', right_on='PATIENT')
dfz.to_csv(model_data_path+'/encounter_pivot_test.csv')
