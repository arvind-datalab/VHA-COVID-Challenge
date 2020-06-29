import pandas as pd
import numpy as np

#Declaring the path

train_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/train'
test_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/test'
model_data_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/model_data'



#------ Train Data
pat = pd.read_csv(model_data_path+"\demographics_train.csv")
immun = pd.read_csv(train_path+"\immunizations.csv")

#Data manipulation
#Conditions for the correct patient set and required dates
pat1 = pat[(pat['covid_flag'] == 1) | (pat['covid_flag'] == 0)]
immun_pat = pd.merge(pat1,immun,left_on='Id',right_on='PATIENT')
immun_pat.Index_date=pd.to_datetime(immun_pat.Index_date)
immun_pat.DATE=pd.to_datetime(immun_pat.DATE)

immun_rqd = immun_pat[(immun_pat.Index_date-immun_pat.DATE).dt.days>=21]

immun_rqd = immun_rqd[['PATIENT', 'DESCRIPTION', 'covid_flag']]
immun2 = immun_rqd.drop_duplicates()
immun_wide = immun2.pivot_table(index=('PATIENT','covid_flag'),columns='DESCRIPTION', aggfunc='size',fill_value=0).reset_index();
immun_wide.to_csv(model_data_path+"\Immunization_pivot_train.csv", index=False)



#----- Test Data
pat_test = pd.read_csv(model_data_path+"\demographics_test.csv")
immun_test = pd.read_csv(test_path+"\immunizations.csv")

#Data manipulation
#Conditions for the correct patient set and required dates
immun_pat = pd.merge(pat_test,immun_test,left_on='Id',right_on='PATIENT')
immun_rqd = immun_pat[['PATIENT', 'DESCRIPTION']]
immun2 = immun_rqd.drop_duplicates()
immun_wide = immun2.pivot_table(index=('PATIENT'),columns='DESCRIPTION', aggfunc='size',fill_value=0).reset_index();
immun_wide.to_csv(model_data_path+"\Immunization_pivot_test.csv", index=False)