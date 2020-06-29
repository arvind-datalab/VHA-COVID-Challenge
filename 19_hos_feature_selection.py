import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

#--Path
train_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/train'
test_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/test'
model_data_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/model_data'
output_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Output'

car_df= pd.read_csv(model_data_path+'/car_data.csv')
regression= pd.read_csv(model_data_path+'/regression.csv')

car_df=car_df[(car_df['inpatient_flag'] == 1)]

df2= car_df.drop(['icu_date','inpatient_date','LOS_ICU','vent_date','covid_flag','PAYERBUCKET_NO COVERAGE',
                  'PAYERBUCKET_MED','PAYERBUCKET_HIGH','PAYERBUCKET_LOW','Cause of Death [US Standard Certificate of Death]',
                  'vent_flag','icu_flag','deceased_flag'],axis=1)
df3=df2.drop(regression.Regression, axis=1)
#---Variable Selection Code
data=df3
data=data.drop('Id',axis=1)
x=pd.DataFrame(data.iloc[:,data.columns!='LOS_HOS'])
y=pd.DataFrame(data.iloc[:,data.columns=='LOS_HOS'])
x=x.fillna(0)
y=y.fillna(0)
X_TRAIN,X_TEST,y_TRAIN,y_TEST=train_test_split(x,y,test_size=0.3,random_state=10)
trainX=X_TRAIN.fillna(0)
testX=X_TEST.fillna(0)
trainy=y_TRAIN.fillna(0)
testy=y_TEST.fillna(0)

model = GradientBoostingRegressor(n_estimators=200,learning_rate=0.1,criterion='mse',max_depth=4,loss='ls')
#model = ExtraTreesClassifier(n_estimators=100, random_state=0)
model.fit(trainX,trainy.values.ravel())

model_score = model.score(trainX,trainy.values.ravel())
# Have a look at R sq to give an idea of the fit ,
# Explained variance score: 1 is perfect prediction
print('R2 sq: ',model_score)
y_pred = model.predict(testX)

# The mean squared error
print("Root Mean squared error: %.2f"% sqrt(mean_squared_error(testy, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Test Variance score: %.2f' % r2_score(testy, y_pred))

# So let's run the model against the test data
from sklearn.model_selection import cross_val_predict

fig, ax = plt.subplots()
ax.scatter(testy, y_pred, edgecolors=(0, 0, 0))
ax.plot([testy.min(), testy.max()], [testy.min(), testy.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Ground Truth vs Predicted")
plt.show()
# For Displaying Top 10 Feature list

feat_importances = pd.Series(model.feature_importances_,index=x.columns)
feat_importances.nlargest(10).plot(kind='barh')
feat_imp = list(zip(x.columns,model.feature_importances_))
feat_imp = pd.DataFrame(feat_imp,columns=["feature_name","importance"]).sort_values('importance' ,ascending = False)
feat_imp.to_csv(output_path+'/variable_imp_hos.csv',index=False)


