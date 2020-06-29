import pandas as pd
import numpy as np

#Declaring the path

train_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/train'
test_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/test'
model_data_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/model_data'

# Train Data
pat = pd.read_csv(model_data_path+"\demographics_train.csv")
devices = pd.read_csv(train_path+"\devices.csv")

#Data manipulation
#Conditions for the correct patient set and required dates
pat1 = pat[(pat['covid_flag'] == 1) | (pat['covid_flag'] == 0)]
devices_pat = pd.merge(pat1,devices,left_on='Id',right_on='PATIENT')
devices_pat.Index_date=pd.to_datetime(devices_pat.Index_date)
devices_pat.START=pd.to_datetime(devices_pat.START)
devices_pat.STOP=pd.to_datetime(devices_pat.STOP)
devices_rqd = devices_pat[(devices_pat.Index_date-devices_pat.START).dt.days>=21 & (devices_pat.STOP>devices_pat.Index_date) | (devices_pat.STOP == 'NaT')]

devices_rqd = devices_rqd[['PATIENT', 'DESCRIPTION', 'covid_flag']]
devices2 = devices_rqd.drop_duplicates()
devices_wide = devices2.pivot_table(index=('PATIENT','covid_flag'),columns='DESCRIPTION', aggfunc='size',fill_value=0).reset_index();
devices_wide.to_csv(model_data_path+"\Devices_pivot_train.csv", index=False)

# Test Data
pat_test = pd.read_csv(model_data_path+"\demographics_test.csv")
devices_test = pd.read_csv(test_path+"\devices.csv")

#Data manipulation
#Conditions for the correct patient set and required dates
devices_pat = pd.merge(pat_test,devices_test,left_on='Id',right_on='PATIENT')
devices_rqd = devices_pat[['PATIENT', 'DESCRIPTION']]
devices2 = devices_rqd.drop_duplicates()
devices_wide = devices2.pivot_table(index=('PATIENT'),columns='DESCRIPTION', aggfunc='size',fill_value=0).reset_index();
devices_wide.to_csv(model_data_path+"\Devices_pivot_test.csv", index=False)
