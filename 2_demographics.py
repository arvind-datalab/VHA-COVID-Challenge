import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
#--Path
train_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/train'
test_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/test'
model_data_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/model_data'


#---train data
patient_df= pd.read_csv(train_path+'/patients.csv')
covid= pd.read_csv(model_data_path+'/covid_patient_ids.csv')
no_covid= pd.read_csv(model_data_path+'/negative_covid_patient_ids.csv')
deceased= pd.read_csv(model_data_path+'/deceased_ids.csv')
icu= pd.read_csv(model_data_path+'/icu_ids.csv')
inpatient= pd.read_csv(model_data_path+'/inpatient_ids.csv')
vent= pd.read_csv(model_data_path+'/vent_ids.csv')
all_p= pd.read_csv(model_data_path+'/all_ids.csv')


df3=pd.merge(patient_df, covid, how='left', left_on='Id', right_on='covid_id')
df3=pd.merge(df3, no_covid, how='left', left_on='Id', right_on='no_covid_id')
df3=pd.merge(df3, deceased, how='left', left_on='Id', right_on='deceased_id')
df3=pd.merge(df3, icu, how='left', left_on='Id', right_on='icu_id')
df3=pd.merge(df3, inpatient, how='left', left_on='Id', right_on='inpatient_id')
df3=pd.merge(df3, vent, how='left', left_on='Id', right_on='vent_id')
df3=pd.merge(df3, all_p, how='left', left_on='Id', right_on='all_id')

df3['vent_flag'] = np.where(pd.notnull(df3['vent_id']), 1, 0)
df3['inpatient_flag'] = np.where(pd.notnull(df3['inpatient_id']), 1, 0)
df3['icu_flag'] = np.where(pd.notnull(df3['icu_id']), 1, 0)
df3['deceased_flag'] = np.where(pd.notnull(df3['deceased_id']), 0, 1)
df3['covid_flag'] = np.where(pd.notnull(df3['covid_id']),1, np.where(pd.notnull(df3['no_covid_id']), 0, 0))
df3['Index_date'] = np.where(pd.notnull(df3['covid_id']), df3['covid_date'], np.where(pd.notnull(df3['no_covid_id']), df3['no_covid_date'], df3['last_ecn_date']))
df3['covid_flag'].value_counts()
df3['Index_date']=pd.to_datetime(df3['Index_date'])
df4=df3.drop(['all_id','covid_date','covid_id','deceased_id','icu_id','inpatient_id','last_ecn_date',
              'no_covid_date','no_covid_id','Unnamed: 0','Unnamed: 0_x','Unnamed: 0_y','vent_id'], axis=1)

#-------Test data
patient_test= pd.read_csv(test_path+'/patients.csv')

#------Demographic code funtion
def demographic(data,name):
    from datetime import datetime
    now = datetime.now()
    data2=data.drop(['SSN','DRIVERS','PASSPORT','PREFIX','FIRST','LAST','SUFFIX','MAIDEN','BIRTHPLACE','ADDRESS','CITY','STATE','COUNTY','ZIP','LAT','LON'], axis=1)
    data2['BIRTHYEAR'] = data2.BIRTHDATE.str[:4].astype(int)
    data2['DEATHYEAR'] = data2.DEATHDATE.str[:4].replace(np.nan, now.year).astype(int)
    data2['AGE'] = data2['DEATHYEAR']-data2['BIRTHYEAR']
    bins = [0, 15, 30, 45, 60, 75, 120]
    labels = ['0-14', '15-29', '30-44', '45-59', '60-75', '75+']
    data2['AGE_GROUP'] = pd.cut(data2['AGE'], bins, labels = labels,include_lowest = True)
    #----- Define expense group
    bins_h = [5.166400e+02, 6.093919e+05, 1.467964e+06, 3.203008e+06]
    # 0-25% ---- 25-75% --- 75%+
    labels_h = ['LOW', 'MED', 'HIGH']
    data2['HEALTHCARE_EXPENSES_GROUP'] = pd.cut(data2['HEALTHCARE_EXPENSES'], bins_h, labels = labels_h,include_lowest = True)
        
    #----- Define expense group
    bins_c = [0, 0.2, 0.8, 11]
    # 0-25% ---- 25-75% --- 75%+
    labels_c = ['LOW', 'MED', 'HIGH']
    data2['COVERAGE_PCT'] = data2['HEALTHCARE_COVERAGE']/data2['HEALTHCARE_EXPENSES']
    data2['COVERAGE_PCT'] = pd.cut(data2['COVERAGE_PCT'], bins_c, labels = labels_c,include_lowest = True)
    data2['COVERAGE_PCT'].value_counts()
    df5=data2.drop(['BIRTHYEAR','DEATHYEAR','AGE','HEALTHCARE_EXPENSES','BIRTHDATE','HEALTHCARE_COVERAGE'], axis=1)
    df5=pd.get_dummies(data=df5, columns=['MARITAL', 'RACE','GENDER','HEALTHCARE_EXPENSES_GROUP','AGE_GROUP','COVERAGE_PCT','ETHNICITY'])
    df5.to_csv(model_data_path+'/demographics_'+name+'.csv')
    
#------ Call Funtion
demographic(df4,'train')
demographic(patient_test,'test')