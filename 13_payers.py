import pandas as pd
import numpy as np

#Declaring the path


#train_path = r'C:\Users\devika.vijayan.s\Accenture\Arvind, Chauhan - VA Challenge\Dataset\train'
#out_path = r'C:\Users\devika.vijayan.s\Accenture\Arvind, Chauhan - VA Challenge\Dataset\model_data'

train_path = 'C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/train'
test_path = 'C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/test'
out_path = 'C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/model_data'


#------ Create train Data

pat = pd.read_csv(out_path+"\demographics_train.csv")
payer = pd.read_csv(train_path+"\payers.csv")
encounter = pd.read_csv(train_path+"\encounters.csv")
payer_t = pd.read_csv(train_path+"\payer_transitions.csv")

encounters_df1 = encounter[['PATIENT','START','STOP','PAYER','TOTAL_CLAIM_COST','PAYER_COVERAGE']]
patient_df1 = pat[['Id','covid_flag','Index_date']]
payer_df1 = payer[['Id','NAME']]

payer_enc = pd.merge(encounters_df1,payer_df1,left_on='PAYER',right_on='Id')
payer_enc = pd.merge(payer_enc,patient_df1,how='inner',left_on='PATIENT',right_on='Id')
payer_enc = payer_enc[(payer_enc['covid_flag'] == 1) | (payer_enc['covid_flag'] == 0)]

payer_enc['START']=pd.to_datetime(payer_enc['START'], format='%d/%m/%Y')
payer_enc['Index_date']=pd.to_datetime(payer_enc['Index_date'], format='%d/%m/%Y')
payer_rqd = payer_enc[(payer_enc['Index_date']-payer_enc['START']).dt.days>=21]

#----- Calculate coverage
payer_agg = payer_rqd[['PATIENT','PAYER_COVERAGE','TOTAL_CLAIM_COST']]
payer_agg = payer_agg.groupby(['PATIENT'], as_index=False).agg({"PAYER_COVERAGE": "sum","TOTAL_CLAIM_COST": "sum"})
payer_agg['COVERAGE_PCT']= (payer_agg['PAYER_COVERAGE'])/(payer_agg['TOTAL_CLAIM_COST'])
bins_p = [-10, 0.00001, 0.5, 0.80, 11]
labels_p = ['NO COVERAGE','LOW', 'MED', 'HIGH']
payer_agg["PAYERBUCKET"] = pd.cut(payer_agg['COVERAGE_PCT'], bins=bins_p,labels=labels_p)

payer_agg=pd.get_dummies(data=payer_agg, columns=['PAYERBUCKET'])
payer_agg=payer_agg.drop(['PAYER_COVERAGE', 'TOTAL_CLAIM_COST'], axis=1)
payer_agg = payer_agg.drop_duplicates()

#--- Caculate latest payer
payer_name=pd.merge(payer_t,payer_df1,left_on='PAYER',right_on='Id')
payer_year = payer_name.groupby(['PATIENT'], as_index=False).agg({"END_YEAR": "max"})
payer_name2=pd.merge(payer_name,payer_year,left_on=['PATIENT', 'END_YEAR'],right_on=['PATIENT', 'END_YEAR'])
payer_name2=payer_name2.drop(['START_YEAR', 'END_YEAR','Id','PAYER','OWNERSHIP'], axis=1)
payer_name2=pd.get_dummies(data=payer_name2, columns=['NAME'])
payer_name2 = payer_name2.drop_duplicates()

payer_wide=pd.merge(payer_name2,payer_agg,how='right',left_on='PATIENT',right_on='PATIENT')
payer_wide=payer_wide.fillna(0)
payer_wide.to_csv(out_path+"\Payers_pivot_train.csv", index=False)


#------ Create yest Data

payer_test = pd.read_csv(test_path+"\payers.csv")
encounter_test = pd.read_csv(test_path+"\encounters.csv")
payer_t_test = pd.read_csv(test_path+"\payer_transitions.csv")

encounters_df1 = encounter_test[['PATIENT','START','STOP','PAYER','TOTAL_CLAIM_COST','PAYER_COVERAGE']]
payer_df1 = payer_test[['Id','NAME']]

payer_enc = pd.merge(encounters_df1,payer_df1,left_on='PAYER',right_on='Id')

#----- Calculate coverage
payer_agg = payer_enc[['PATIENT','PAYER_COVERAGE','TOTAL_CLAIM_COST']]
payer_agg = payer_agg.groupby(['PATIENT'], as_index=False).agg({"PAYER_COVERAGE": "sum","TOTAL_CLAIM_COST": "sum"})
payer_agg['COVERAGE_PCT']= (payer_agg['PAYER_COVERAGE'])/(payer_agg['TOTAL_CLAIM_COST'])
bins_p = [-10, 0.00001, 0.5, 0.80, 11]
labels_p = ['NO COVERAGE','LOW', 'MED', 'HIGH']
payer_agg["PAYERBUCKET"] = pd.cut(payer_agg['COVERAGE_PCT'], bins=bins_p,labels=labels_p)
payer_agg["PAYERBUCKET"].value_counts()
payer_agg=pd.get_dummies(data=payer_agg, columns=['PAYERBUCKET'])
payer_agg=payer_agg.drop(['PAYER_COVERAGE', 'TOTAL_CLAIM_COST'], axis=1)
payer_agg = payer_agg.drop_duplicates()

#--- Caculate latest payer
payer_name=pd.merge(payer_t_test,payer_df1,left_on='PAYER',right_on='Id')
payer_year = payer_name.groupby(['PATIENT'], as_index=False).agg({"END_YEAR": "max"})
payer_name2=pd.merge(payer_name,payer_year,left_on=['PATIENT', 'END_YEAR'],right_on=['PATIENT', 'END_YEAR'])
payer_name2=payer_name2.drop(['START_YEAR', 'END_YEAR','Id','PAYER','OWNERSHIP'], axis=1)
payer_name2=pd.get_dummies(data=payer_name2, columns=['NAME'])
payer_name2 = payer_name2.drop_duplicates()

payer_wide=pd.merge(payer_name2,payer_agg,how='right',left_on='PATIENT',right_on='PATIENT')
payer_wide=payer_wide.fillna(0)
payer_wide.to_csv(out_path+"\Payers_pivot_test.csv", index=False)