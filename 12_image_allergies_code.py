import pandas as pd
import numpy as np
from datetime import datetime, timedelta

train_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/train'
test_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/test'
model_data_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/model_data'

#---- Train data
image_data = pd.read_csv(train_path+"/imaging_studies.csv")
allergies = pd.read_csv(train_path+"/allergies.csv")
demographics_train = pd.read_csv(model_data_path+"/demographics_train.csv")

demographics_train1 = demographics_train[(demographics_train['covid_flag'] == 1) | (demographics_train['covid_flag'] == 0)]
image_data['Body_part_scan'] = image_data['BODYSITE_DESCRIPTION'] + "_" + image_data['SOP_DESCRIPTION']
image_int1 = pd.merge(demographics_train1,image_data,left_on=['Id'],right_on = ['PATIENT'],how = 'inner')
image_int1.Index_date=pd.to_datetime(image_int1.Index_date)
image_int1.DATE=pd.to_datetime(image_int1.DATE)
image_int2 = image_int1[(image_int1['DATE'] >= (image_int1['Index_date'] - timedelta(days=21))) & (image_int1['DATE'] <= image_int1['Index_date'])]
image_int2 ['flag'] = 1
pivoted_image = image_int2 .pivot_table(values = 'flag',index=['PATIENT'], columns='Body_part_scan',aggfunc = np.max, fill_value = 0).reset_index()
pivoted_image.fillna(0,inplace=True)

allergies_int1 = pd.merge(demographics_train1,allergies,left_on=['Id'],right_on = ['PATIENT'],how = 'inner')
allergies_int1.Index_date=pd.to_datetime(allergies_int1.Index_date)
allergies_int1.START=pd.to_datetime(allergies_int1.START)
allergies_int2 = allergies_int1[(allergies_int1['START'] <= (allergies_int1['Index_date'] - timedelta(days=21)))]
allergies_int2['flag'] = 1
pivoted_allergies = allergies_int2.pivot_table(values = 'flag',index=['PATIENT'], columns='DESCRIPTION',aggfunc = np.max, fill_value = 0).reset_index()
pivoted_allergies.fillna(0,inplace=True)

pivoted_allergies.to_csv(model_data_path+"/allergies_data.csv")
pivoted_image.to_csv(model_data_path+"/images_data.csv")


#---- Test Data
image_data_test = pd.read_csv(test_path+"/imaging_studies.csv")
allergies_test = pd.read_csv(test_path+"/allergies.csv")
demographics_test = pd.read_csv(model_data_path+"/demographics_test.csv")

image_data_test['Body_part_scan'] = image_data_test['BODYSITE_DESCRIPTION'] + "_" + image_data_test['SOP_DESCRIPTION']
image_int1 = pd.merge(demographics_test,image_data_test,left_on=['Id'],right_on = ['PATIENT'],how = 'inner')
image_int1 ['flag'] = 1
pivoted_image = image_int1.pivot_table(values = 'flag',index=['PATIENT'], columns='Body_part_scan',aggfunc = np.max, fill_value = 0).reset_index()
pivoted_image.fillna(0,inplace=True)

allergies_int1 = pd.merge(demographics_train1,allergies,left_on=['Id'],right_on = ['PATIENT'],how = 'inner')
allergies_int1['flag'] = 1
pivoted_allergies = allergies_int1.pivot_table(values = 'flag',index=['PATIENT'], columns='DESCRIPTION',aggfunc = np.max, fill_value = 0).reset_index()
pivoted_allergies.fillna(0,inplace=True)

pivoted_allergies.to_csv(model_data_path+"/allergies_test.csv")
pivoted_image.to_csv(model_data_path+"/images_test.csv")