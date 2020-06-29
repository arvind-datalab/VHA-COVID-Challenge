import pandas as pd
import numpy as np

#Declaring the path

train_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/train'
test_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/test'
model_data_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/model_data'


# Train Data
#Import required dataframes
pat = pd.read_csv(model_data_path+"\demographics_train.csv")
care = pd.read_csv(train_path+"\careplans.csv")

#Data manipulation
#Conditions for the correct patient set and required dates
pat1 = pat[(pat['covid_flag'] == 1) | (pat['covid_flag'] == 0)]
care_pat = pd.merge(pat1,care,left_on='Id',right_on='PATIENT')
care_pat.Index_date=pd.to_datetime(care_pat.Index_date)
care_pat.START=pd.to_datetime(care_pat.START)
care_pat.STOP=pd.to_datetime(care_pat.STOP)
care_rqd = care_pat[((care_pat.Index_date-care_pat.START).dt.days>=21) & (care_pat.STOP>care_pat.Index_date) | (care_pat.STOP == 'NaT')]
care_rqd = care_rqd[['PATIENT', 'DESCRIPTION', 'covid_flag']]
care2 = care_rqd.drop_duplicates()
care_wide = care2.pivot_table(index=('PATIENT','covid_flag'),columns='DESCRIPTION', aggfunc='size',fill_value=0).reset_index();
care_wide.to_csv(model_data_path+"\Careplans_pivot_train.csv", index=False)

# Test Data
#Import required dataframes
pat_test = pd.read_csv(model_data_path+"\demographics_test.csv")
care_test = pd.read_csv(test_path+"\careplans.csv")

#Data manipulation
#Conditions for the correct patient set and required dates
care_pat = pd.merge(pat_test,care_test,left_on='Id',right_on='PATIENT')
care_rqd = care_pat[['PATIENT', 'DESCRIPTION']]
care2 = care_rqd.drop_duplicates()
care_wide = care2.pivot_table(index=('PATIENT'),columns='DESCRIPTION', aggfunc='size',fill_value=0).reset_index();
care_wide.to_csv(model_data_path+"\Careplans_pivot_test.csv", index=False)