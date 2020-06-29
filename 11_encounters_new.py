import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#--Path
#train_path='C:/Users/priyanka.s.iyer/Accenture/Arvind, Chauhan - VA Challenge/Dataset/train'
#test_path='C:/Users/priyanka.s.iyer/Accenture/Arvind, Chauhan - VA Challenge/Dataset/test'
#model_data_path='C:/Users/priyanka.s.iyer/Accenture/Arvind, Chauhan - VA Challenge/Dataset/model_data'

train_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/train'
test_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/test'
model_data_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/model_data'

#---Test Dataframes
encounters_train= pd.read_csv(train_path+'/encounters.csv')
patient_train= pd.read_csv(model_data_path+'/demographics_train.csv')

encounters_df1 = encounters_train[['PATIENT','START', 'ENCOUNTERCLASS','STOP','DESCRIPTION','REASONDESCRIPTION','CODE']]
encounters_df1['TYPE_DESC']=encounters_df1['ENCOUNTERCLASS']+" "+encounters_df1['REASONDESCRIPTION']
encounters_df1['TYPE_DESC']=np.where(encounters_df1['ENCOUNTERCLASS'] == 'inpatient', 
              (encounters_df1['ENCOUNTERCLASS']+encounters_df1['REASONDESCRIPTION']),'')
patient_df1 = patient_train[['Id','covid_flag','Index_date']]
df1=pd.merge(patient_df1, encounters_df1, how='inner', left_on='Id', right_on='PATIENT')
df1['Index_date']=pd.to_datetime(df1['Index_date'])
df1['START'] = pd.to_datetime(df1['START']).dt.strftime('%d/%m/%Y')
df1['STOP'] = pd.to_datetime(df1['STOP']).dt.strftime('%d/%m/%Y')
df2=df1[(df1.Index_date-df1.START).dt.days>=21]

#------ LOS variables
df1.START = pd.to_datetime(df1.START)
df1.STOP= pd.to_datetime(df1.STOP)
df1['LOS']=(df1.STOP-df1.START).dt.days


##------------------------------------------------ INPATIENT STAY DAYS ---------------------------------------------#
#------------Replace outliers
LOS2=pd.DataFrame(df1[(df1['LOS'] >= 1)&(df1['ENCOUNTERCLASS'] == 'inpatient') & df1['TYPE_DESC'].notnull()]['LOS'])

def outlier_treatment(datacolumn):
 sorted(datacolumn)
 Q1,Q3 = np.percentile(datacolumn , [25,99])
 IQR = Q3 - Q1
 lower_range = 0
# upper_range = Q3 + (5 * IQR)
 upper_range=60
 return lower_range,upper_range

range=outlier_treatment(LOS2)
range #(0, 8.5)
range[0] 
range[1]
min(df1.LOS) #0
max(df1.LOS) #28700

#Capping Outliers 
df2=df1['LOS_cleaned']=np.where(df1['LOS']>range[1], range[1],np.where(df1['LOS']<range[0], range[0],df1['LOS']))
df2=pd.DataFrame(df1[(df1['LOS_cleaned'] >= 1)&(df1['ENCOUNTERCLASS'] == 'inpatient') & df1['TYPE_DESC'].notnull()])
df2=df2[['Id', 'TYPE_DESC', 'LOS_cleaned']]
df3=df2.groupby(['Id', 'TYPE_DESC'])['LOS_cleaned'].sum().reset_index()
encounters_ip_los_pivot=df2.pivot_table(index='Id', columns='TYPE_DESC',  values='LOS_cleaned')
encounters_ip_los_pivot
#Export 
#encounters_ip_los_pivot.to_csv(model_data_path+'/encounters_ip_los_pivot.csv')

#--------------------------------------------------- AVG STAY DAYS FOR ENC --------------------------------------------#

df4=df1['LOS_cleaned']=np.where(df1['LOS']>range[1], range[1],np.where(df1['LOS']<range[0], range[0],df1['LOS']))
df4=pd.DataFrame(df1[(df1['LOS_cleaned'] >= 1)])
df4=df4[['Id', 'ENCOUNTERCLASS', 'LOS_cleaned']]

df5=df4.groupby(['Id', 'ENCOUNTERCLASS']).agg({'LOS_cleaned' :'sum'})
df5.columns=['TOT_LOS'] 
df5_pivot=df5.pivot_table(index='Id', columns='ENCOUNTERCLASS',  values='TOT_LOS').reset_index()
df5_pivot2=df5_pivot.drop(['urgentcare','wellness'],axis=1)
df5_pivot2.columns=['Id','ambulatory_TOT_LOS', 'emergency_TOT_LOS', 'inpatient_TOT_LOS', 'outpatient_TOT_LOS']

df6=df4.groupby(['Id', 'ENCOUNTERCLASS']).agg({'ENCOUNTERCLASS' : 'count'})
df6.columns=['TOT_VISITS'] 
df6_pivot=df6.pivot_table(index='Id', columns='ENCOUNTERCLASS',  values='TOT_VISITS')
df6_pivot2=df6_pivot.drop(['urgentcare','wellness'],axis=1)

encounter_agg = pd.merge(df5_pivot2, df6_pivot2, how='left', on='Id')
#---------------------------------------------------------Merging data-------------------------------------------#

encounter_continuous = pd.merge(encounters_ip_los_pivot, encounter_agg, how='right', on='Id' )
bins = [0, 1, 3, 5, 100]
labels = ['0-1', '1-3', '3-5', '5+']
encounter_continuous['inpatient_visit_bucket'] = pd.cut(encounter_continuous['inpatient'], bins, labels = labels,include_lowest = True)

bins2 = [0, 1, 7, 15, 200]
labels2 = ['0-1', '1-7', '7-15', '15+']
encounter_continuous['inpatient_TOT_LOS_bucket'] = pd.cut(encounter_continuous['inpatient_TOT_LOS'], bins2, labels = labels2,include_lowest = True)

encounter_continuous=pd.get_dummies(data=encounter_continuous, columns=['inpatient_visit_bucket', 'inpatient_TOT_LOS_bucket'])
    

encounter_continuous.to_csv(model_data_path+'/encounter_continuous_train.csv')


#---Test Dataframes
encounters_test= pd.read_csv(test_path+'/encounters.csv')
patient_test= pd.read_csv(test_path+'/patients.csv')

encounters_df1 = encounters_test[['PATIENT','START', 'ENCOUNTERCLASS','STOP','DESCRIPTION','REASONDESCRIPTION','CODE']]
encounters_df1['TYPE_DESC']=encounters_df1['ENCOUNTERCLASS']+" "+encounters_df1['REASONDESCRIPTION']
encounters_df1['TYPE_DESC']=np.where(encounters_df1['ENCOUNTERCLASS'] == 'inpatient', 
              (encounters_df1['ENCOUNTERCLASS']+encounters_df1['REASONDESCRIPTION']),'')
patient_df1 = patient_test[['Id']]
df1=pd.merge(patient_df1, encounters_df1, how='inner', left_on='Id', right_on='PATIENT')

#------ LOS variables
df1.START = pd.to_datetime(df1.START)
df1.STOP= pd.to_datetime(df1.STOP)
df1['LOS']=(df1.STOP-df1.START).dt.days


##------------------------------------------------ INPATIENT STAY DAYS ---------------------------------------------#
#------------Replace outliers
LOS2=pd.DataFrame(df1[(df1['LOS'] >= 1)&(df1['ENCOUNTERCLASS'] == 'inpatient') & df1['TYPE_DESC'].notnull()]['LOS'])

def outlier_treatment(datacolumn):
 sorted(datacolumn)
 Q1,Q3 = np.percentile(datacolumn , [25,99])
 IQR = Q3 - Q1
 lower_range = 0
# upper_range = Q3 + (5 * IQR)
 upper_range=60
 return lower_range,upper_range

range=outlier_treatment(LOS2)
range #(0, 8.5)
range[0] 
range[1]
min(df1.LOS) #0
max(df1.LOS) #28700

#Capping Outliers 
df2=df1['LOS_cleaned']=np.where(df1['LOS']>range[1], range[1],np.where(df1['LOS']<range[0], range[0],df1['LOS']))
df2=pd.DataFrame(df1[(df1['LOS_cleaned'] >= 1)&(df1['ENCOUNTERCLASS'] == 'inpatient') & df1['TYPE_DESC'].notnull()])
df2=df2[['Id', 'TYPE_DESC', 'LOS_cleaned']]
df3=df2.groupby(['Id', 'TYPE_DESC'])['LOS_cleaned'].sum().reset_index()
encounters_ip_los_pivot=df2.pivot_table(index='Id', columns='TYPE_DESC',  values='LOS_cleaned')


#--------------------------------------------------- AVG STAY DAYS FOR ENC --------------------------------------------#

df4=df1['LOS_cleaned']=np.where(df1['LOS']>range[1], range[1],np.where(df1['LOS']<range[0], range[0],df1['LOS']))
df4=pd.DataFrame(df1[(df1['LOS_cleaned'] >= 1)])
df4=df4[['Id', 'ENCOUNTERCLASS', 'LOS_cleaned']]

df5=df4.groupby(['Id', 'ENCOUNTERCLASS']).agg({'LOS_cleaned' :'sum'})
df5.columns=['TOT_LOS'] 
df5_pivot=df5.pivot_table(index='Id', columns='ENCOUNTERCLASS',  values='TOT_LOS')
df5_pivot.columns=['ambulatory_TOT_LOS', 'emergency_TOT_LOS', 'inpatient_TOT_LOS', 'outpatient_TOT_LOS']

df6=df4.groupby(['Id', 'ENCOUNTERCLASS']).agg({'ENCOUNTERCLASS' : 'count'})
df6.columns=['TOT_VISITS'] 
df6_pivot=df6.pivot_table(index='Id', columns='ENCOUNTERCLASS',  values='TOT_VISITS')

encounter_agg = pd.merge(df5_pivot, df6_pivot, how='left', on='Id')
#---------------------------------------------------------Merging data-------------------------------------------#

encounter_continuous = pd.merge(encounters_ip_los_pivot, encounter_agg, how='right', on='Id' )
bins = [0, 1, 3, 5, 100]
labels = ['0-1', '1-3', '3-5', '5+']
encounter_continuous['inpatient_visit_bucket'] = pd.cut(encounter_continuous['inpatient'], bins, labels = labels,include_lowest = True)

bins2 = [0, 1, 7, 15, 200]
labels2 = ['0-1', '1-7', '7-15', '15+']
encounter_continuous['inpatient_TOT_LOS_bucket'] = pd.cut(encounter_continuous['inpatient_TOT_LOS'], bins2, labels = labels2,include_lowest = True)

encounter_continuous=pd.get_dummies(data=encounter_continuous, columns=['inpatient_visit_bucket', 'inpatient_TOT_LOS_bucket'])
   
encounter_continuous.to_csv(model_data_path+'/encounter_continuous_test.csv')
