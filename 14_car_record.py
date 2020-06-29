import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#--Path
train_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/train'
test_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/test'
model_data_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/model_data'
output_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Output'


#------- Train Car Record
patient_df= pd.read_csv(model_data_path+'/demographics_train.csv')
observations_df= pd.read_csv(model_data_path+'/observation_pivot_train.csv')
procedure_df= pd.read_csv(model_data_path+'/procedure_pivot_train.csv')
medication_df= pd.read_csv(model_data_path+'/medication_pivot_train.csv')
careplans_df= pd.read_csv(model_data_path+'/Careplans_pivot_train.csv')
immunization_df= pd.read_csv(model_data_path+'/Immunization_pivot_train.csv')
devices_df= pd.read_csv(model_data_path+'/Devices_pivot_train.csv')
finding_df= pd.read_csv(model_data_path+'/Finding.csv')
disorder_df= pd.read_csv(model_data_path+'/Disorder.csv')
allergies_df= pd.read_csv(model_data_path+'/allergies_data.csv')
images_df= pd.read_csv(model_data_path+'/images_data.csv')
encounter_df= pd.read_csv(model_data_path+'/encounter_continuous_train.csv')
payer_df= pd.read_csv(model_data_path+'/Payers_pivot_train.csv')


patient_df=patient_df[(patient_df['covid_flag'] == 0) | (patient_df['covid_flag'] == 1)]
observations_df= observations_df.drop(['covid_flag'],axis=1)
procedure_df= procedure_df.drop(['covid_flag'],axis=1)
medication_df= medication_df.drop(['covid_flag'],axis=1)
careplans_df= careplans_df.drop(['covid_flag'],axis=1)
immunization_df= immunization_df.drop(['covid_flag'],axis=1)
devices_df= devices_df.drop(['covid_flag'],axis=1)
#encounter_df= encounter_df.drop(['covid_flag'],axis=1)
disorder_df= disorder_df.drop(['covid_flag'],axis=1)
finding_df= finding_df.drop(['covid_flag'],axis=1)

#--CAR Record creation
df1=pd.merge(patient_df, observations_df, how='left', left_on='Id', right_on='PATIENT')
df1=pd.merge(df1, procedure_df, how='left', left_on='Id', right_on='PATIENT')
df1=pd.merge(df1, medication_df, how='left', left_on='Id', right_on='PATIENT')
df1=pd.merge(df1, careplans_df, how='left', left_on='Id', right_on='PATIENT')
df1=pd.merge(df1, immunization_df, how='left', left_on='Id', right_on='PATIENT')
df1=pd.merge(df1, devices_df, how='left', left_on='Id', right_on='PATIENT')
df1=pd.merge(df1, finding_df, how='left', left_on='Id', right_on='PATIENT')
df1=pd.merge(df1, disorder_df, how='left', left_on='Id', right_on='PATIENT')
df1=pd.merge(df1, allergies_df, how='left', left_on='Id', right_on='PATIENT')
df1=pd.merge(df1, images_df, how='left', left_on='Id', right_on='PATIENT')
df1=pd.merge(df1, encounter_df, how='left', left_on='Id', right_on='Id')
df1=pd.merge(df1, payer_df, how='left', left_on='Id', right_on='PATIENT')
df2= df1.drop(['Unnamed: 0_x','Unnamed: 0_y','PATIENT_x','PATIENT_y','Unnamed: 0'],axis=1)

df3=pd.DataFrame(df2.columns)
df3.to_csv(model_data_path+'/col.csv')


#-------- Test CAR record

patient_test= pd.read_csv(model_data_path+'/demographics_test.csv')
observations_test= pd.read_csv(model_data_path+'/observation_pivot_test.csv')
procedure_test= pd.read_csv(model_data_path+'/procedure_pivot_test.csv')
medication_test= pd.read_csv(model_data_path+'/medication_pivot_test.csv')
careplans_test= pd.read_csv(model_data_path+'/Careplans_pivot_test.csv')
immunization_test= pd.read_csv(model_data_path+'/Immunization_pivot_test.csv')
devices_test= pd.read_csv(model_data_path+'/Devices_pivot_test.csv')
disorder_test= pd.read_csv(model_data_path+'/condition_test.csv')
allergies_test= pd.read_csv(model_data_path+'/allergies_test.csv')
images_test= pd.read_csv(model_data_path+'/images_test.csv')
encounter_test= pd.read_csv(model_data_path+'/encounter_continuous_test.csv')
payer_test= pd.read_csv(model_data_path+'/Payers_pivot_test.csv')


#--CAR Record creation
df4=pd.merge(patient_test, observations_test, how='left', left_on='Id', right_on='PATIENT')
df4=pd.merge(df4, procedure_test, how='left', left_on='Id', right_on='PATIENT')
df4=pd.merge(df4, medication_test, how='left', left_on='Id', right_on='PATIENT')
df4=pd.merge(df4, careplans_test, how='left', left_on='Id', right_on='PATIENT')
df4=pd.merge(df4, immunization_test, how='left', left_on='Id', right_on='PATIENT')
df4=pd.merge(df4, devices_test, how='left', left_on='Id', right_on='PATIENT')
df4=pd.merge(df4, disorder_test, how='left', left_on='Id', right_on='PATIENT')
df4=pd.merge(df4, allergies_test, how='left', left_on='Id', right_on='PATIENT')
df4=pd.merge(df4, images_test, how='left', left_on='Id', right_on='PATIENT')
df4=pd.merge(df4, encounter_test, how='left', left_on='Id', right_on='Id')
df4=pd.merge(df4, payer_test, how='left', left_on='Id', right_on='PATIENT')
df5= df4.drop(['Unnamed: 0_x','DEATHDATE','PATIENT_x','PATIENT_y','Unnamed: 0','Unnamed: 0_y'],axis=1)
df5.to_csv(model_data_path+'/car_data_test.csv',index=False)
df6=pd.DataFrame(df5.columns)
df6.to_csv(model_data_path+'/col_test.csv')

#-------- Filter variable from train data set who are not present in test
final_train_var=pd.merge(df3,df6,how='inner', on=0)
final_train_var = final_train_var.append({0 : 'LOS_ICU'} , ignore_index=True)
final_train_var = final_train_var.append({0 : 'icu_date'} , ignore_index=True)
final_train_var = final_train_var.append({0 : 'inpatient_date'} , ignore_index=True)
final_train_var = final_train_var.append({0 : 'LOS_HOS'} , ignore_index=True)
final_train_var = final_train_var.append({0 : 'vent_date'} , ignore_index=True)
final_train_var = final_train_var.append({0 : 'vent_flag'} , ignore_index=True)
final_train_var = final_train_var.append({0 : 'inpatient_flag'} , ignore_index=True)
final_train_var = final_train_var.append({0 : 'icu_flag'} , ignore_index=True)
final_train_var = final_train_var.append({0 : 'deceased_flag'} , ignore_index=True)
final_train_var = final_train_var.append({0 : 'covid_flag'} , ignore_index=True)
df2=df2.filter(final_train_var[0])
df2.to_csv(model_data_path+'/car_data.csv',index=False)
df7=pd.DataFrame(df2.columns)
df7.to_csv(model_data_path+'/col.csv')