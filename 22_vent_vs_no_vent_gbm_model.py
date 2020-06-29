import pandas as pd
import numpy as np
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#--Path
train_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/train'
test_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/test'
model_data_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/model_data'
output_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Output'

car_df= pd.read_csv(model_data_path+'/car_data.csv')
features_df= pd.read_csv(output_path+'/variable_imp_vent.csv')
car_test= pd.read_csv(model_data_path+'/car_data_test.csv')

def topFeatures(Df_source,top_percent):
    maxval =Df_source['importance'].shape[0]
    top_pos=0
    for i in range(maxval):
        x_top=Df_source.nlargest(i,['importance'])
        if x_top['importance'].sum() >=top_percent:
            top_pos=i
            break
    topFeatureslist=pd.DataFrame(Df_source.nlargest(top_pos,['importance'])['feature_name'])
    return(topFeatureslist)

top_feature_df=topFeatures(features_df,0.90)
top_feature_df1=top_feature_df
top_feature_df1 = top_feature_df1.append({'feature_name' : 'Id'} , ignore_index=True)
top_feature_df2 = top_feature_df1.append({'feature_name' : 'vent_flag'} , ignore_index=True)
car_top_df=car_df.filter(top_feature_df2['feature_name'])


#---Model Code
data=car_top_df
data=data.drop('Id',axis=1)
x=pd.DataFrame(data.iloc[:,data.columns!='vent_flag'])
y=pd.DataFrame(data.iloc[:,data.columns=='vent_flag'])
x=x.fillna(0)
y=y.fillna(0)
X_TRAIN,X_TEST,y_TRAIN,y_TEST=train_test_split(x,y,test_size=0.3,random_state=10)
trainX=X_TRAIN.fillna(0)
testX=X_TEST.fillna(0)
trainy=y_TRAIN.fillna(0)
testy=y_TEST.fillna(0)
#smt=SMOTE()
#trainX,trainy=smt.fit_sample(trainX,trainy)
# Feature importance
model = GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,max_depth=7)
model.fit(trainX,trainy.values.ravel())
y_pred=model.predict(testX)
acc_score_gb=accuracy_score(testy,y_pred)
cm_gb=confusion_matrix(testy,y_pred)
cl_gb=classification_report(testy,y_pred)
precision, recall, thresholds = precision_recall_curve(testy, y_pred)
roc=roc_auc_score(testy,y_pred)
roc_curve=roc_curve(testy,y_pred) 

print(cm_gb)
print(acc_score_gb)
print(cl_gb)
print(precision)
print(recall)

#------ Final Variable List
feat_importances = pd.Series(model.feature_importances_,index=x.columns)
feat_importances.nlargest(10).plot(kind='barh')
feat_imp = list(zip(x.columns,model.feature_importances_))
feat_imp = pd.DataFrame(feat_imp,columns=["feature_name","importance"])
feat_imp.to_csv(output_path+'/variable_imp_final_vent.csv')

# generate feature importance count
car_top_df_melt=car_top_df.melt(id_vars =['Id','vent_flag']) 
car_top_df_count=pd.DataFrame(car_top_df_melt.groupby(['variable','vent_flag'])['value'].agg(np.sum)).reset_index()
car_top_df_pivot=car_top_df_count.pivot_table(index=('variable'),columns='vent_flag',aggfunc='sum',fill_value=0).reset_index().rename(columns={"Index": "Index4", 0: "deceased_id"})
car_top_df_pivot.columns = ['variable', 'no_vent', 'vent']
car_top_df_pivot['total_patient']=car_top_df_pivot['no_vent']+car_top_df_pivot['vent']

# for Final Output
car_idcnt_ft=pd.merge(left=feat_imp,right=car_top_df_pivot,left_on='feature_name',right_on='variable').sort_values('importance' ,ascending = False)
car_idcnt_ft2=car_idcnt_ft.drop(['variable'],axis=1)
# To get the Patient Count by Segment
car_idcnt_ft2.to_csv(output_path+'/variable_summary_final_vent.csv')

#--------------------------- Model Output

car_top_test = car_test.filter(top_feature_df1['feature_name'])
car_top_test1=car_top_test.drop(['Id'],axis=1)
car_top_test1=car_top_test1.fillna(0) 
y_pred_test=pd.DataFrame(model.predict_proba(car_top_test1))

pd.set_option('precision',8)
pred_new = pd.DataFrame(y_pred_test)
result = pd.DataFrame(data={"Id":car_top_test['Id'],"prediction":pred_new[1]}) 

covid_output= pd.read_csv(output_path+'/covid_no_covid_output_input.csv')
covid_output['covid_flag']=np.where(covid_output['prediction']>=0.5,1,0)
covid_output_rename=covid_output.rename(columns={"Id": "Id", "prediction":"covid_prob", "covid_flag":"covid_flag"})
vent_output=pd.merge(covid_output_rename,result , how='left', on ='Id')
vent_output['prediction_new']=np.where(vent_output['covid_flag']==1,vent_output['prediction'],0)
final_result=vent_output.drop(['covid_prob','covid_flag','prediction'],axis=1)
final_result.to_csv(output_path+'/vent_no_vent_output.csv', index = False,header=False)
