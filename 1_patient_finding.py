# -*- coding: utf-8 -*-
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
#--Path
train_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/train'
test_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/test'
model_data_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/model_data'


#---Dataframes
patient_df= pd.read_csv(train_path+'/patients.csv')
condition_df= pd.read_csv(train_path+'/conditions.csv')
observation_df= pd.read_csv(train_path+'/observations.csv')
encounters_df= pd.read_csv(train_path+'/encounters.csv')
procedure_df= pd.read_csv(train_path+'/procedures.csv')

#------Length of Stay
encounters_df.START=pd.to_datetime(encounters_df.START)
encounters_df.STOP=pd.to_datetime(encounters_df.STOP)
 
encounters_df['LOS_HOS'] = np.where(((encounters_df['REASONCODE'] == 840539006) & (encounters_df['CODE'] == 1505002)) , (encounters_df.STOP-encounters_df.START).dt.days, 0)
encounters_df['LOS_ICU'] = np.where(((encounters_df['CODE'] == 305351004)), (encounters_df.STOP-encounters_df.START).dt.days, 0)


#------ Covid Patient
covid_patient_ids=condition_df[condition_df['CODE'] == 840539006].PATIENT
covid_patient_id=condition_df[condition_df['CODE'] == 840539006][['PATIENT','START']].reset_index(drop=True).rename(columns={"PATIENT": "covid_id", "START": "covid_date"})
covid_patient_id.covid_date=pd.to_datetime(covid_patient_id.covid_date, format='%d/%m/%Y')
covid_patient_id.to_csv(model_data_path+'/covid_patient_ids.csv')


#------ No-Covid Patient
negative_covid_patient_ids=observation_df[(observation_df['CODE'] == '94531-1')&(observation_df['VALUE']=='Not detected (qualifier value)')][['PATIENT','DATE']].reset_index(drop=True)
negative_covid_patient_ids_2 = negative_covid_patient_ids.groupby(['PATIENT'], as_index=False).agg({"DATE": "min"}).rename(columns={"DATE": "no_covid_date","PATIENT": "no_covid_id"})
negative_covid_patient_ids_2.no_covid_date=pd.to_datetime(negative_covid_patient_ids_2.no_covid_date, format='%d/%m/%Y')
negative_covid_patient_ids_2.to_csv(model_data_path+'/negative_covid_patient_ids.csv')

#------ Hospitalized Patient
inpatient_ids=pd.DataFrame(encounters_df[(encounters_df['REASONCODE'] == 840539006)&(encounters_df['CODE'] == 1505002)][['PATIENT','START','LOS_HOS']]).rename(columns={"START": "inpatient_date", "PATIENT": "inpatient_id"})
inpatient_ids.inpatient_date=pd.to_datetime(inpatient_ids.inpatient_date, format='%d/%m/%Y')
inpatient_ids.to_csv(model_data_path+'/inpatient_ids.csv')

#------ Deceased Patient
deceased_ids=pd.DataFrame(np.intersect1d(covid_patient_ids,patient_df[patient_df['DEATHDATE'].notna()].Id)).rename(columns={"Index": "Index4", 0: "deceased_id"})
deceased_ids.to_csv(model_data_path+'/deceased_ids.csv')

#------ Vent Patient
vent_ids=procedure_df[(procedure_df['CODE'] == 26763009)&(procedure_df['PATIENT'].isin(covid_patient_ids))][['PATIENT','DATE']].reset_index(drop=True)
vent_ids_2 = vent_ids.groupby(['PATIENT'], as_index=False).agg({"DATE": "min"}).rename(columns={"DATE": "vent_date","PATIENT": "vent_id"})
vent_ids_2.vent_date=pd.to_datetime(vent_ids_2.vent_date, format='%d/%m/%Y')
vent_ids_2.to_csv(model_data_path+'/vent_ids.csv')

#------ ICU Patient
icu_ids=encounters_df[(encounters_df['CODE'] == 305351004)&(encounters_df['PATIENT'].isin(covid_patient_ids))][['PATIENT','START','LOS_ICU']].reset_index(drop=True)
icu_ids_2 = icu_ids.groupby(['PATIENT','LOS_ICU'], as_index=False).agg({"START": "min"}).rename(columns={"START": "icu_date","PATIENT": "icu_id"})
icu_ids_2.icu_date=pd.to_datetime(icu_ids_2.icu_date, format='%d/%m/%Y')
icu_ids_2.to_csv(model_data_path+'/icu_ids.csv')


#------ All Patient encounter
all_ids=encounters_df[['PATIENT','START']].reset_index(drop=True)
all_ids_2 = all_ids.groupby(['PATIENT'], as_index=False).agg({"START": "max"}).rename(columns={"START": "last_ecn_date","PATIENT": "all_id"})
all_ids_2.last_ecn_date=pd.to_datetime(all_ids_2.last_ecn_date, format='%d/%m/%Y')
all_ids_2.to_csv(model_data_path+'/all_ids.csv')

