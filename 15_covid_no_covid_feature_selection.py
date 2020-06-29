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

car_df= pd.read_csv(model_data_path+'/car_data.csv')
classifier= pd.read_csv(model_data_path+'/classifier.csv')


car_df=car_df[(car_df['covid_flag'] == 0) | (car_df['covid_flag'] == 1)]

df2= car_df.drop(['LOS_ICU','icu_date','inpatient_date','LOS_HOS','vent_date'
                  ,'vent_flag','inpatient_flag','icu_flag','deceased_flag','COVERAGE_PCT'],axis=1)
df3=df2.drop(classifier.classifier, axis=1)

#---Variable Selection Code
data=df3
data=data.drop('Id',axis=1)
x=pd.DataFrame(data.iloc[:,data.columns!='covid_flag'])
y=pd.DataFrame(data.iloc[:,data.columns=='covid_flag'])
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
model = GradientBoostingClassifier(n_estimators=200,learning_rate=0.1,max_depth=4)
#model = ExtraTreesClassifier(n_estimators=100, random_state=0)
model.fit(trainX,trainy.values.ravel())
y_pred=model.predict(testX)
acc_score_gb=accuracy_score(testy,y_pred)
cm_gb=confusion_matrix(testy,y_pred)
cl_gb=classification_report(testy,y_pred)
print(cm_gb)
print(acc_score_gb)
print(cl_gb)

# For Displaying Top 10 Feature list

feat_importances = pd.Series(model.feature_importances_,index=x.columns)
feat_importances.nlargest(10).plot(kind='barh')
feat_imp = list(zip(x.columns,model.feature_importances_))
feat_imp = pd.DataFrame(feat_imp,columns=["feature_name","importance"])
feat_imp.to_csv(output_path+'/variable_imp_covid.csv',index=False)

# generate feature importance count
df2_melt=df2.melt(id_vars =['Id','covid_flag']) 
df2_count=pd.DataFrame(df2_melt.groupby(['variable','covid_flag'])['value'].agg(np.sum)).reset_index()
df2_pivot=df2_count.pivot_table(index=('variable'),columns='covid_flag',aggfunc='sum',fill_value=0).reset_index().rename(columns={"Index": "Index4", 0: "deceased_id"})
df2_pivot.columns = ['variable', 'no_covid', 'covid']
df2_pivot['total_patient']=df2_pivot['no_covid']+df2_pivot['covid']

# for Final Output
var_idcnt_ft=pd.merge(left=feat_imp,right=df2_pivot,left_on='feature_name',right_on='variable').sort_values('importance' ,ascending = False)
var_idcnt_ft2=var_idcnt_ft.drop(['variable'],axis=1)
# To get the Patient Count by Segment
var_idcnt_ft2.to_csv(output_path+'/variable_summary_cycle1.csv',index=False)

