import pandas as pd
import numpy as np
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt


#--Path
train_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/train'
test_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/test'
model_data_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Dataset/model_data'
output_path='C:/Users/chauhan.arvind/OneDrive - Accenture/10. VA/VA Challenge/Output'

car_df= pd.read_csv(model_data_path+'/car_data.csv')
features_df= pd.read_csv(output_path+'/variable_imp_icu.csv')
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

top_feature_df=topFeatures(features_df,0.95)
top_feature_df1=top_feature_df
top_feature_df1 = top_feature_df1.append({'feature_name' : 'Id'} , ignore_index=True)
top_feature_df2 = top_feature_df1.append({'feature_name' : 'LOS_ICU'} , ignore_index=True)
car_top_df=car_df.filter(top_feature_df2['feature_name'])

#---Model Code
data=car_top_df
data=data.drop('Id',axis=1)
x=pd.DataFrame(data.iloc[:,data.columns!='LOS_ICU'])
y=pd.DataFrame(data.iloc[:,data.columns=='LOS_ICU'])
x=x.fillna(0)
y=y.fillna(0)
X_TRAIN,X_TEST,y_TRAIN,y_TEST=train_test_split(x,y,test_size=0.3,random_state=10)
trainX=X_TRAIN.fillna(0)
testX=X_TEST.fillna(0)
trainy=y_TRAIN.fillna(0)
testy=y_TEST.fillna(0)
#model = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1)

# Regression models for comparison
#models = [GradientBoostingRegressor(random_state = 0), 
#          LinearRegression(),
#          KNeighborsRegressor(),
#          RandomForestRegressor(random_state = 0)]
#
#results = {}
#
#for model in models:
#    # Instantiate and fit Regressor Model
#    reg_model = model
#    reg_model.fit(trainX, trainy.values.ravel())
#    # Make predictions with model
#    y_test_preds = reg_model.predict(testX)
#    # Grab model name and store results associated with model
#    name = str(model).split("(")[0]
#    results[name] = r2_score(testy, y_test_preds)
#    print('{} done.'.format(name))

model=GradientBoostingRegressor(n_estimators=500,learning_rate=0.1,criterion='mse',max_depth=7,loss='ls')

model_score = model.fit(trainX,trainy.values.ravel())
model_score = model.score(trainX,trainy.values.ravel())
# Have a look at R sq to give an idea of the fit ,
# Explained variance score: 1 is perfect prediction
print('R2 sq: ',model_score)
y_pred = model.predict(testX)



# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(testy, y_pred))
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
feat_imp = pd.DataFrame(feat_imp,columns=["feature_name","importance"])
feat_imp.to_csv(output_path+'/final_variable_list_icu.csv')


#--------------------------- VAlidation Output

car_top_test = car_test.filter(top_feature_df1['feature_name'])
car_top_test1=car_top_test.drop(['Id'],axis=1)
car_top_test1=car_top_test1.fillna(0) 
y_pred_test=pd.DataFrame(model.predict(car_top_test1))

pd.set_option('precision',8)
pred_new = pd.DataFrame(y_pred)
result = pd.DataFrame(data={"Id":car_top_test['Id'],"prediction":pred_new[0]}) 

covid_output= pd.read_csv(output_path+'/covid_no_covid_output_input.csv')
covid_output['covid_flag']=np.where(covid_output['prediction']>=0.5,1,0)
covid_output_rename=covid_output.rename(columns={"Id": "Id", "prediction":"covid_prob", "covid_flag":"covid_flag"})
icu_output=pd.merge(covid_output_rename,result , how='left', on ='Id')
icu_output['prediction_new']=np.where(icu_output['covid_flag']==1,icu_output['prediction'],0)
final_result=icu_output.drop(['covid_prob','covid_flag','prediction'],axis=1)
final_result.to_csv(output_path+'/icu_output.csv', index = False,header=False)
