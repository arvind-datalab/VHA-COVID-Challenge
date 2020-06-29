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

train_path='C:/Users/anup.b.sharma/Accenture/Arvind, Chauhan - VA Challenge/Dataset/train'
test_path='C:/Users/anup.b.sharma/Accenture/Arvind, Chauhan - VA Challenge/Dataset/test'
model_data_path='C:/Users/anup.b.sharma/Accenture/Arvind, Chauhan - VA Challenge/Dataset/model_data'
output_path='C:/Users/anup.b.sharma/Accenture/Arvind, Chauhan - VA Challenge/Output'

car_df= pd.read_csv(model_data_path+'/car_data.csv')
features_df= pd.read_csv(output_path+'/variable_imp_alive.csv')
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
top_feature_df2 = top_feature_df1.append({'feature_name' : 'deceased_flag'} , ignore_index=True)
car_top_df=car_df.filter(top_feature_df2['feature_name'])


#---Model Code
data=car_top_d
data.head()
data=data.drop('Id',axis=1)
x=pd.DataFrame(data.iloc[:,data.columns!='deceased_flag'])
y=pd.DataFrame(data.iloc[:,data.columns=='deceased_flag'])
x=x.fillna(0)
y=y.fillna(0)
X_TRAIN,X_TEST,y_TRAIN,y_TEST=train_test_split(x,y,test_size=0.3,random_state=10)
trainX=X_TRAIN.fillna(0)
testX=X_TEST.fillna(0)
trainy=y_TRAIN.fillna(0)
testy=y_TEST.fillna(0)
smt=SMOTE()

trainX.head()
trainX,trainy=smt.fit_sample(trainX,trainy.values.ravel())
# Feature importance
model = GradientBoostingClassifier(n_estimators=50,learning_rate=0.001)
model.fit(trainX,trainy)
history = model.fit(trainX,trainy)
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

plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()


# model Keras
trainX = np.array(trainX)
trainy = np.array(trainy)

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.metrics import categorical_crossentropy

model_dec = Sequential([Dense(258, input_shape=(258,), activation='relu'),
                    Dense(100, activation='sigmoid'),
                    Dense(1, activation='sigmoid')])

model_dec.compile('sgd', loss='binary_crossentropy', metrics=['accuracy'])

history=model_dec.fit(trainX, trainy, validation_split=0.2, batch_size=1000, epochs=100, shuffle=True, verbose=2)

testX = np.array(testX)
testy = np.array(testy)
predictions = model_dec.predict(testX, batch_size=100, verbose=0)
prediction_classes1 = model_dec.predict_classes(testX, batch_size=100, verbose=0)

predictions


acc_score_gb=accuracy_score(testy,prediction_classes1)
acc_score_k=accuracy_score(testy,prediction_classes1)
cm_k=confusion_matrix(testy,prediction_classes1)
cl_k=classification_report(testy,prediction_classes1)
precision, recall, thresholds = precision_recall_curve(testy, prediction_classes1)
roc=roc_auc_score(testy,prediction_classes1)
roc_curve=roc_curve(testy,prediction_classes1) 


# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

print(cm_k)
print(acc_score_k)
print(cl_k)
print(precision)
print(recall)



#------ Final Variable List
feat_importances = pd.Series(model.feature_importances_,index=x.columns)
feat_importances.nlargest(10).plot(kind='barh')
feat_imp = list(zip(x.columns,model.feature_importances_))
feat_imp = pd.DataFrame(feat_imp,columns=["feature_name","importance"])
feat_imp.to_csv(output_path+'/variable_imp_final_alive.csv')

# generate feature importance count
car_top_df_melt=car_top_df.melt(id_vars =['Id','deceased_flag']) 
car_top_df_count=pd.DataFrame(car_top_df_melt.groupby(['variable','deceased_flag'])['value'].agg(np.sum)).reset_index()
car_top_df_pivot=car_top_df_count.pivot_table(index=('variable'),columns='deceased_flag',aggfunc='sum',fill_value=0).reset_index().rename(columns={"Index": "Index4", 0: "deceased_id"})
car_top_df_pivot.columns = ['variable', 'death', 'alive']
car_top_df_pivot['total_patient']=car_top_df_pivot['death']+car_top_df_pivot['alive']

# for Final Output
car_idcnt_ft=pd.merge(left=feat_imp,right=car_top_df_pivot,left_on='feature_name',right_on='variable').sort_values('importance' ,ascending = False)
car_idcnt_ft2=car_idcnt_ft.drop(['variable'],axis=1)
# To get the Patient Count by Segment
car_idcnt_ft2.to_csv(output_path+'/variable_summary_final_alive.csv')

#--------------------------- Model Output
car_top_test = car_test.filter(top_feature_df1['feature_name'])
car_top_test1=car_top_test.drop(['Id'],axis=1)
car_top_test1=car_top_test1.fillna(0) 
y_pred_test=pd.DataFrame(model_dec.predict(car_top_test1))

pd.set_option('precision',8)
pred_new = pd.DataFrame(y_pred_test)
result = pd.DataFrame(data={"Id":car_top_test['Id'],"prediction":pred_new[0]}) 

covid_output= pd.read_csv(output_path+'/covid_no_covid_output_input.csv')
covid_output['covid_flag']=np.where(covid_output['prediction']>=0.5,1,0)
covid_output_rename=covid_output.rename(columns={"Id": "Id", "prediction":"covid_prob", "covid_flag":"covid_flag"})
alive_output=pd.merge(covid_output_rename,result , how='left', on ='Id')
alive_output['prediction_new']=np.where(alive_output['covid_flag']==1,alive_output['prediction'],1)
final_result=alive_output.drop(['covid_prob','covid_flag','prediction'],axis=1)
final_result.to_csv(output_path+'/alive_output.csv', index = False,header=False)
