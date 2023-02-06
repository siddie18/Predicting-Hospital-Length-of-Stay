#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[2]:


adm_data = pd.read_csv('C:\\Users\\Siddi\\OneDrive\\Documents\\MIMIC Datasets\\mimic-iii-clinical-database-1.4\\ADMISSIONS.csv\\ADMISSIONS.csv')
diag_data = pd.read_csv('C:\\Users\\Siddi\\OneDrive\\Documents\\MIMIC Datasets\\mimic-iii-clinical-database-1.4\\DIAGNOSES_ICD.csv\\DIAGNOSES_ICD.csv')
icu_data=pd.read_csv('C:\\Users\\Siddi\\OneDrive\\Documents\\MIMIC Datasets\\mimic-iii-clinical-database-1.4\\ICUSTAYS.csv\\ICUSTAYS.csv')
pat_data=pd.read_csv('C:\\Users\\Siddi\\OneDrive\\Documents\\MIMIC Datasets\\mimic-iii-clinical-database-1.4\\PATIENTS.csv\\PATIENTS.csv')


# In[3]:


diag_data.head()


# In[4]:


adm_data.head()


# In[5]:


pat_data.head()


# In[6]:


icu_data.head()


# In[7]:


print(adm_data.info())
print("-------------------------------------------------------------")
print(icu_data.info())
print("-------------------------------------------------------------")
print(pat_data.info())
print("-------------------------------------------------------------")
print(diag_data.info())


# In[8]:


print(adm_data.shape)
print("-------------------------------------------------------------")
print(icu_data.shape)
print("-------------------------------------------------------------")
print(pat_data.shape)
print("-------------------------------------------------------------")
print(diag_data.shape)


# In[9]:


print(adm_data.describe())
print("-------------------------------------------------------------")
print(icu_data.describe())
print("-------------------------------------------------------------")
print(pat_data.describe())
print("-------------------------------------------------------------")
print(diag_data.describe())


# In[10]:


# EDA and feature engineering in admission csv


# In[11]:


adm_data.head()


# In[12]:


adm_data.ADMISSION_TYPE.value_counts().plot(kind="bar")
plt.show()


# In[13]:


# we can see most the admission type is of emergency 


# In[14]:


# according to my analysis emergency is equal to urgent 


# In[15]:


adm_data.ADMISSION_TYPE=adm_data.ADMISSION_TYPE.apply(lambda x: str(x).replace("URGENT","EMERGENCY"))


# In[16]:


adm_data.ADMISSION_TYPE.value_counts(normalize=True).plot(kind="bar")
plt.show()


# In[17]:


adm_data.ADMISSION_LOCATION.value_counts()


# In[18]:


adm_data.ADMISSION_LOCATION.value_counts().plot(kind="bar")
plt.show()


# In[19]:


# we will combine all the transfers into one 


# In[20]:


z=["TRANSFER FROM HOSP/EXTRAM","TRANSFER FROM SKILLED NUR","TRANSFER FROM OTHER HEALT","TRSF WITHIN THIS FACILITY"]


# In[21]:


for i in z:
    adm_data.ADMISSION_LOCATION=adm_data.ADMISSION_LOCATION.apply(lambda x: str(x).replace(i,"transfer from other facilities"))
    


# In[22]:


adm_data.ADMISSION_LOCATION.value_counts()


# In[23]:


# we can also previous admissions are of refferal, converting them refferal from facilties 


# In[24]:


y=["PHYS REFERRAL/NORMAL DELI","CLINIC REFERRAL/PREMATURE","HMO REFERRAL/SICK"]


# In[25]:


for i in y:
    adm_data.ADMISSION_LOCATION=adm_data.ADMISSION_LOCATION.apply(lambda x: str(x).replace(i,"refferal from other facilities"))
    


# In[26]:


adm_data.ADMISSION_LOCATION.value_counts().plot(kind="bar")
plt.show()


# In[27]:


## we can see most of the admission location are emergency room admit 


# In[28]:


adm_data.DISCHARGE_LOCATION.value_counts()


# In[29]:


adm_data.DISCHARGE_LOCATION.value_counts().plot(kind="bar")
plt.show()


# In[30]:


# HOME WITH HOME IV PROVIDR and HOSPICE home is also a HOME with health care so replacing its value


# In[31]:


adm_data.DISCHARGE_LOCATION=adm_data.DISCHARGE_LOCATION.apply(lambda x: str(x).replace("HOME WITH HOME IV PROVIDR","HOME HEALTH CARE"))


# In[32]:


adm_data.DISCHARGE_LOCATION=adm_data.DISCHARGE_LOCATION.apply(lambda x: str(x).replace("HOSPICE-HOME","HOME HEALTH CARE"))


# In[33]:


# moreover REHAB/DISTINCT PART HOSP,LONG TERM CARE HOSPITAL, SHORT TERM HOSPITAL, DISC-TRAN CANCER/CHLDRN H , 
#DISCH-TRAN TO PSYCH HOSP etc are sent to some other facilities 


# In[34]:


a=["SNF","REHAB/DISTINCT PART HOSP","LONG TERM CARE HOSPITAL","SHORT TERM HOSPITAL","DISC-TRAN CANCER/CHLDRN H ","DISCH-TRAN TO PSYCH HOSP","HOSPICE-MEDICAL FACILITY","OTHER FACILITY","ICF","DISC-TRAN TO FEDERAL HC","SNF-MEDICAID ONLY CERTIF","DISC-TRAN CANCER/CHLDRN H"]


# In[35]:


for i in a:
    adm_data.DISCHARGE_LOCATION=adm_data.DISCHARGE_LOCATION.apply(lambda x: str(x).replace(i,"sent to other facilities"))
    


# In[36]:


adm_data.DISCHARGE_LOCATION.value_counts()


# In[37]:


adm_data=adm_data[adm_data['DISCHARGE_LOCATION']!="sent to other facilities-MEDICAID ONLY CERTIF"]
 


# In[38]:


adm_data.DISCHARGE_LOCATION.value_counts().plot(kind="bar")
plt.show()


# In[39]:


adm_data.INSURANCE.value_counts()


# In[40]:


# medicare and mediaid are services offered by government hence converting them


# In[41]:


x=["Medicare","Medicaid"]


# In[42]:


for i in x:
    adm_data.INSURANCE=adm_data.INSURANCE.apply(lambda x: str(x).replace(i,"Government"))
    


# In[43]:


adm_data.INSURANCE.value_counts().plot(kind="bar")
plt.show()


# In[44]:


# we wont require language column as it doesnt seems perfect for what we are predicting


# In[45]:


adm_data.drop(["LANGUAGE"], axis = 1, inplace=True)


# In[46]:


# we wont require religion column as it doesnt seems perfect for what we are predicting


# In[47]:


adm_data.drop(["RELIGION"], axis = 1, inplace=True)


# In[48]:


adm_data.MARITAL_STATUS.value_counts()


# In[49]:


# separted is same as divorce and life partner is same as married also converting unknown to np.nan


# In[50]:


adm_data.MARITAL_STATUS=adm_data.MARITAL_STATUS.apply(lambda x: str(x).replace("LIFE PARTNER","MARRIED"))
adm_data.MARITAL_STATUS=adm_data.MARITAL_STATUS.apply(lambda x: str(x).replace("SEPARATED","DIVORCED"))


# In[51]:


adm_data["MARITAL_STATUS"].replace("UNKNOWN (DEFAULT)", np.nan, inplace=True)
adm_data["MARITAL_STATUS"].replace("nan", np.nan, inplace=True)


# In[52]:


adm_data.MARITAL_STATUS.value_counts().plot(kind="bar")
plt.show()


# In[53]:


adm_data.ETHNICITY.value_counts()


# In[54]:


v=["ASIAN - CAMBODIAN","ASIAN - OTHER","ASIAN - KOREAN","ASIAN - JAPANESE","ASIAN - THAI","ASIAN - VIETNAMESE","ASIAN - ASIAN INDIAN","ASIAN - CHINESE","ASIAN - FILIPINO"]


# In[55]:


for i in v:
    adm_data.ETHNICITY=adm_data.ETHNICITY.apply(lambda x: str(x).replace(i,"ASIAN"))
    


# In[56]:


adm_data.ETHNICITY.replace(regex=r'^HISPANIC\D*', value='HISPANIC/LATINO', inplace=True)


# In[57]:


adm_data.ETHNICITY.replace(regex=r'^WHITE\D*', value='WHITE', inplace=True)


# In[58]:


adm_data.ETHNICITY.replace(regex=r'^BLACK\D*', value='BLACK/AFRICAN AMERICAN', inplace=True)


# In[59]:


t=["UNABLE TO OBTAIN", "OTHER", "PATIENT DECLINED TO ANSWER","UNKNOWN/NOT SPECIFIED"]


# In[60]:


adm_data['ETHNICITY'].replace(t, value='Unknown/other', inplace=True)


# In[61]:


adm_data.ETHNICITY.loc[~adm_data.ETHNICITY.isin(adm_data.ETHNICITY.value_counts().nlargest(5).index.tolist())] = 'Unknown/other'


# In[62]:


adm_data.ETHNICITY.value_counts().plot(kind="bar")
plt.show()


# In[63]:


adm_data.DIAGNOSIS.value_counts()


# In[64]:


adm_data.HOSPITAL_EXPIRE_FLAG.value_counts().plot(kind="bar")
plt.show()


# In[65]:


adm_data['HOSPITAL_EXPIRE_FLAG']=adm_data['HOSPITAL_EXPIRE_FLAG'].astype(object)


# In[66]:


adm_data.HAS_CHARTEVENTS_DATA.value_counts().plot(kind="bar")
plt.show()


# In[67]:


adm_data.ADMITTIME = pd.to_datetime(adm_data.ADMITTIME)
adm_data.DISCHTIME = pd.to_datetime(adm_data.DISCHTIME)


# In[68]:


adm_data.head()


# In[69]:


adm_data["LEN_OF_STAY"]= round((adm_data.DISCHTIME-adm_data.ADMITTIME).dt.total_seconds()/86400,3)


# In[70]:


adm_data.columns


# In[71]:


adm_data.head()


# In[72]:


adm_data.LEN_OF_STAY.describe()


# In[73]:


# length of stay has negative values. We have to remove them


# In[74]:


adm_data=adm_data[adm_data.LEN_OF_STAY>0]


# In[75]:


# dropping records in which patients were died because they were never discharged 


# In[76]:


adm__data=adm_data[adm_data.HOSPITAL_EXPIRE_FLAG==0]


# In[77]:


sns.distplot(adm__data.LEN_OF_STAY)
plt.show()


# In[78]:


# we can length of stay is right skewed 


# In[79]:


# list of columns no longer required 

m=["ROW_ID","DISCHTIME","DEATHTIME","EDREGTIME","EDOUTTIME","HAS_CHARTEVENTS_DATA","HOSPITAL_EXPIRE_FLAG"]


# In[80]:


adm__data.drop(m, axis=1, inplace=True)


# In[81]:


adm__data.head()


# In[82]:


## EDA and feature engineering of patients data


# In[83]:


pat_data.head()


# In[84]:


# removing all death related columns because we are finding length of stay and these patients were never discharged 
pat_data.drop(["ROW_ID","DOD","DOD_HOSP","DOD_SSN","EXPIRE_FLAG"],axis=1, inplace=True)


# In[85]:


pat_data.head()


# In[86]:


pat_data['DOB'] = pd.to_datetime(pat_data['DOB']).dt.date


# In[87]:


pat_data.head()


# In[88]:


pat_data.GENDER.value_counts(normalize=True).plot(kind="bar")
plt.show()


# In[89]:


## EDA and feature engineering diagnoses data


# In[90]:


diag_data.head()


# In[91]:


diag_data.info()


# In[92]:


diag_data["codetype"]=diag_data.ICD9_CODE
diag_data["codetype"] = diag_data.ICD9_CODE[~diag_data.ICD9_CODE.str.contains("[a-zA-Z]").fillna(False)]


# In[93]:


diag_data.head()


# In[94]:


diag_data["codetype"].fillna(value='999', inplace=True)


# In[95]:


diag_data['codetype'] = diag_data['codetype'].str.slice(start=0, stop=3, step=1)
diag_data['codetype'] = diag_data['codetype'].astype(int)


# In[96]:


code_ranges = [(1, 140), (140, 240), (240, 280), (280, 290), (290, 320), (320, 390),(390, 460), (460, 520), (520, 580), (580, 630), (630, 680), (680, 710),
               (710, 740), (740, 760), (760, 780), (780, 800), (800, 1000), (1000, 2000)]


# In[97]:


diag = {0: 'infectious', 1: 'neoplasms', 2: 'endocrine', 3: 'blood',4: 'mental', 5: 'nervous', 6: 'circulatory', 7: 'respiratory',
        8: 'digestive', 9: 'genitourinary', 10: 'pregnancy', 11: 'skin', 12: 'muscular', 13: 'congenital', 14: 'prenatal', 15: 'misc',
             16: 'injury', 17: 'misc'}


# In[98]:


for i, j in enumerate(code_ranges):
    diag_data['codetype'] = np.where(diag_data['codetype'].between(j[0],j[1]), 
            i,diag_data['codetype'])


# In[99]:


diag_data.head()


# In[100]:


diag_data['codetype'] = diag_data['codetype'].replace(diag)


# In[101]:


diag_data.head()


# In[102]:


diag_data.codetype.value_counts().plot(kind="bar")
plt.show()


# In[103]:


diag = diag_data.groupby('HADM_ID')['codetype'].apply(list).reset_index()
diag.head()


# In[104]:


diag_dummy = pd.get_dummies(diag['codetype'].apply(pd.Series).stack()).sum(level=0)
diag_dummy.head()


# In[105]:


diag_dummy = diag_dummy.join(diag['HADM_ID'], how="outer")
diag_dummy.head()


# In[106]:


diag_dummy.shape


# In[107]:


model_data = adm__data.merge(diag_dummy, how='inner', on='HADM_ID')


# In[108]:


model_data.head()


# In[109]:


model_data = model_data.merge(pat_data, how='inner', on='SUBJECT_ID')


# In[110]:


model_data.head()


# In[111]:


df_age_min = model_data[['SUBJECT_ID', 'ADMITTIME']].groupby('SUBJECT_ID').min().reset_index()
df_age_min.columns = ['SUBJECT_ID', 'ADMIT_MIN']
df_age_min.head()


# In[112]:


model_data= model_data.merge(df_age_min, how='outer', on='SUBJECT_ID')


# In[113]:


model_data['DOB'] = pd.to_datetime(model_data['DOB']).dt.date


# In[114]:


model_data['ADMIT_MIN'] = pd.to_datetime(model_data['ADMIT_MIN']).dt.date


# In[115]:


model_data['age'] = model_data.apply(lambda x: (x['ADMIT_MIN'] - x['DOB']).days/365, axis=1)


# In[116]:


model_data=model_data[model_data.age<100]


# In[117]:


model_data.head()


# In[118]:


icu_data.head()


# In[119]:


icu_data['FIRST_CAREUNIT'].replace({'CCU': 'ICU', 'CSRU': 'ICU', 'MICU': 'ICU',
                                  'SICU': 'ICU', 'TSICU': 'ICU'}, inplace=True)


# In[120]:


icu_data['ICU_flag'] = icu_data['FIRST_CAREUNIT']
icu = icu_data.groupby('HADM_ID')['ICU_flag'].apply(list).reset_index()
icu.head()


# In[121]:


icu_dummy = pd.get_dummies(icu['ICU_flag'].apply(pd.Series).stack()).sum(level=0)
icu_dummy[icu_dummy >= 1] = 1
icu_dummy = icu_dummy.join(icu['HADM_ID'], how="outer")
icu_dummy.head()


# In[122]:


model_data =model_data.merge(icu_dummy, how='outer', on='HADM_ID')


# In[123]:


model_data.ICU.fillna(value=0, inplace=True)
model_data.NICU.fillna(value=0, inplace=True)


# In[124]:


model_data.head()


# In[125]:


model_data.ICU=model_data.ICU.astype("object")


# In[126]:


model_data.NICU=model_data.NICU.astype("object")


# In[127]:


drop=["SUBJECT_ID","HADM_ID","ADMITTIME","DOB","ADMIT_MIN"]


# In[128]:


model_data.drop(drop,axis=1,inplace=True)


# In[129]:


model_data.drop("DIAGNOSIS",axis=1, inplace=True)


# In[130]:


model_data.drop("DISCHARGE_LOCATION",axis=1, inplace=True)


# In[131]:


model_data=pd.get_dummies(model_data, columns=['ADMISSION_TYPE','INSURANCE','ETHNICITY', 'ADMISSION_LOCATION', 'MARITAL_STATUS',"GENDER"])


# In[132]:


model_data.isnull().sum()


# In[133]:


model_data=model_data.dropna()


# In[134]:


model_data.head()


# In[135]:


model_data.info()


# In[ ]:





# In[136]:


model_data


# In[137]:


from sklearn.model_selection import train_test_split


# In[138]:


from sklearn.tree import DecisionTreeRegressor


# In[139]:


dt = DecisionTreeRegressor(random_state=42, max_depth=4, min_samples_leaf=10)


# In[140]:


np.random.seed(0)
df_train, df_test = train_test_split(model_data, train_size=0.8,test_size=0.2, random_state=100,shuffle=True)


# In[141]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[142]:


df_train['LEN_OF_STAY'] = scaler.fit_transform(df_train[['LEN_OF_STAY']])
df_test['LEN_OF_STAY'] = scaler.transform(df_test[['LEN_OF_STAY']])


# In[143]:


y_train = df_train.pop("LEN_OF_STAY")
X_train = df_train

y_test = df_test.pop("LEN_OF_STAY")
X_test = df_test


# In[144]:


X_test.shape, X_train.shape


# In[145]:


dt.fit(X_train, y_train)


# In[146]:


y_train_pred = dt.predict(X_train)


# In[147]:


from sklearn.metrics import r2_score


# In[148]:


r2_score(y_train, y_train_pred)


# In[149]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_train, y_train_pred)


# In[150]:


y_test_pred = dt.predict(X_test)


# In[151]:


r2_score(y_test, y_test_pred)


# In[152]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_test_pred)


# In[153]:


from sklearn.model_selection import GridSearchCV

model = DecisionTreeRegressor()

gs = GridSearchCV(model,
                  param_grid = {'max_depth': range(1, 11),
                                'min_samples_split': range(10, 60, 10)},
                  cv=5,
                  n_jobs=1,
                  scoring='neg_mean_squared_error')

gs.fit(X_train, y_train)

print(gs.best_params_)
print(-gs.best_score_)


# In[154]:


new_model = DecisionTreeRegressor(max_depth=8,
                                  min_samples_split=40)

new_model.fit(X_train, y_train)


# In[155]:


y_train_pred = new_model.predict(X_train)


# In[156]:


r2_score(y_train, y_train_pred)


# In[157]:


mean_squared_error(y_train, y_train_pred)


# In[158]:


y_test_pred = new_model.predict(X_test)


# In[159]:


r2_score(y_test, y_test_pred)


# In[160]:


mean_squared_error(y_test, y_test_pred)


# In[161]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[162]:


#grid_search.fit(X_train, y_train)
#grid_search.best_params_


# In[163]:


rf = RandomForestRegressor(bootstrap= True,max_depth=100,max_features= 3,
                           min_samples_leaf= 3,min_samples_split= 10,n_estimators= 300)


# In[164]:


rf.fit(X_train, y_train)


# In[165]:


y_train_pred = rf.predict(X_train)


# In[166]:


r2_score(y_train, y_train_pred)


# In[167]:


mean_squared_error(y_train, y_train_pred)


# In[168]:


y_test_pred = rf.predict(X_test)


# In[169]:


r2_score(y_test, y_test_pred)


# In[170]:


mean_squared_error(y_test, y_test_pred)


# In[171]:


model_data.head()


# In[172]:


# lets make this an classification problem now 
# if len of stay is less than equal to 10 than its label is 0
# if len of stay is more than 10 than its label is 1


# In[173]:


class_data=model_data.copy()


# In[174]:


class_data.head()


# In[175]:


class_data["class_los_label"]= class_data["LEN_OF_STAY"].apply(lambda x: str(0) if x<=10 else str(1))


# In[176]:


class_data.drop("LEN_OF_STAY", axis=1, inplace=True)


# In[177]:


class_data.class_los_label.value_counts(normalize=True).plot(kind="bar")
plt.show()


# In[178]:


# we can see data data is imbalanced 


# In[179]:


X=class_data.drop(["class_los_label"],axis=1)


# In[180]:


y=class_data["class_los_label"]


# In[181]:


X_train,X_test,y_train,y_test=train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=100, shuffle=True)


# In[182]:


y_train.value_counts(normalize=True)


# In[183]:


y_test.value_counts(normalize=True)


# In[184]:


# handling imbalnce problem 

# SMOTE
from imblearn.over_sampling import SMOTE
smt = SMOTE(random_state=27)
X_resampled_smt, y_resampled_smt = smt.fit_resample(X_train, y_train)
len(X_resampled_smt)


# In[185]:


# Importing decision tree classifier
from sklearn.tree import DecisionTreeClassifier


# In[186]:


dtree = DecisionTreeClassifier()
# Importing RFE
from sklearn.feature_selection import RFE

# Intantiate RFE with 15 columns
rfe = RFE(dtree,n_features_to_select=20)

# Fit the rfe model with train set
rfe = rfe.fit(X_resampled_smt, y_resampled_smt)


# In[187]:


# RFE selected columns
rfe_cols = X_resampled_smt.columns[rfe.support_]
print(rfe_cols)


# In[188]:


# Importing libraries for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
}


# Instantiate the grid search model
dtree = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator = dtree, 
                           param_grid = param_grid, 
                           scoring= 'recall',
                           cv = 5, 
                           verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_resampled_smt[rfe_cols],y_resampled_smt)


# In[189]:


# cv results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results


# In[190]:


print(grid_search.best_estimator_)


# In[191]:


dt= DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, min_samples_split=50)

dt.fit(X_resampled_smt[rfe_cols], y_resampled_smt)


# In[192]:


# Predictions on the train set
y_train_pred = dt.predict(X_resampled_smt[rfe_cols])


# In[193]:


# Confusion matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix
confusion = metrics.confusion_matrix(y_resampled_smt, y_train_pred)
print(confusion)


# In[194]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[195]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_resampled_smt, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[196]:


# Prediction on the test set
y_test_pred = dt.predict(X_test[rfe_cols])


# In[197]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[198]:


P = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[211]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))

#F1 Score
from sklearn.metrics import f1_score
print("F1_Score",f1_score(y_test, y_test_pred, pos_label="0"))


# In[204]:


y_test.astype("str")


# In[212]:


## Logistic regression


# In[213]:


##### Importing stats model
import statsmodels.api as sm


# In[214]:


X_resampled_smt.ICU=X_resampled_smt.ICU.astype("int")


# In[215]:


X_resampled_smt.NICU=X_resampled_smt.NICU.astype("int")


# In[216]:


X_resampled_smt.info()


# In[217]:


y_resampled_smt=y_resampled_smt.astype("int")


# In[218]:


lreg=sm.GLM(y_resampled_smt,(sm.add_constant(X_resampled_smt)),family=sm.families.Binomial())


# In[219]:


# Fit the model
lreg=lreg.fit().summary()


# In[220]:


lreg


# In[221]:


# Importing logistic regression from sklearn
from sklearn.linear_model import LogisticRegression
# Intantiate the logistic regression
logreg = LogisticRegression()


# In[222]:


# Intantiate RFE with 15 columns
rfe = RFE(logreg, n_features_to_select=15)

# Fit the rfe model with train set
rfe = rfe.fit(X_resampled_smt, y_resampled_smt)


# In[223]:


rfe_cols = X_resampled_smt.columns[rfe.support_]
print(rfe_cols)


# In[224]:


# Adding constant to X_train
X_train_sm_1 = sm.add_constant(X_resampled_smt[rfe_cols])

#Instantiate the model
log_1 = sm.GLM(y_resampled_smt, X_train_sm_1, family=sm.families.Binomial())

# Fit the model
log_1 = log_1.fit()

log_1.summary()


# In[225]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[226]:


vif = pd.DataFrame()
vif['Features'] = X_resampled_smt[rfe_cols].columns
vif['VIF'] = [variance_inflation_factor(X_resampled_smt[rfe_cols].values, i) for i in range(X_resampled_smt[rfe_cols].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[227]:


y_train_pred= log_1.predict(X_train_sm_1)
y_train_pred.head()


# In[228]:


y_train_pred_final= pd.DataFrame({'class_label':y_resampled_smt.values, 'prob':y_train_pred.values})

y_train_pred_final.head(10)


# In[229]:


## Finding Optimal Probablity Cutoff Point


# In[230]:


# Creating columns for different probablity cutoffs
prob_cutoff = [float(p/10) for p in range(10)]

for i in prob_cutoff:
    y_train_pred_final[i] = y_train_pred_final['prob'].map(lambda x : 1 if x > i else 0)
    
y_train_pred_final.head()


# In[231]:


# Creating a dataframe
from sklearn import metrics
from sklearn.metrics import confusion_matrix
cutoff_df = pd.DataFrame(columns=['probability', 'accuracy', 'sensitivity', 'specificity'])

for i in prob_cutoff:
    cm1 = metrics.confusion_matrix(y_train_pred_final['class_label'], y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[232]:


# Plotting accuracy, sensitivity and specificity for different probabilities.
cutoff_df.plot('probability', ['accuracy','sensitivity','specificity'])
plt.show()


# In[233]:


# Creating a column with name "predicted", which is the predicted value for 0.6 cutoff 
y_train_pred_final['predicted'] = y_train_pred_final['prob'].map(lambda x: 1 if x > 0.4 else 0)
y_train_pred_final.head()


# In[234]:


# Confusion metrics
confusion = metrics.confusion_matrix(y_train_pred_final['class_label'], y_train_pred_final['predicted'])
print(confusion)


# In[235]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[236]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train_pred_final['class_label'], y_train_pred_final['predicted']))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# In[237]:


# ROC Curve function

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[238]:


draw_roc(y_train_pred_final['class_label'], y_train_pred_final['prob'])


# In[239]:


X_test_log=X_test[rfe_cols]


# In[240]:


X_test_sm = sm.add_constant(X_test_log)


# In[241]:


y_test_pred = log_1.predict(X_test_sm)


# In[242]:


y_test_pred.head()


# In[243]:


y_pred_1 = pd.DataFrame(y_test_pred)
y_pred_1.head()


# In[244]:


# Convetting y_test to a dataframe
y_test_df = pd.DataFrame(y_test)
y_test_df.head()


# In[245]:


y_test_pred_final = pd.concat([y_test_df, y_pred_1], axis=1)


# In[246]:


y_test_pred_final.head()


# In[247]:


y_test_pred_final = y_test_pred_final.rename(columns={0:'prob'})


# In[248]:


y_test_pred_final.head()


# In[249]:


# In the test set using probablity cutoff 0.5, what we got in the train set 
y_test_pred_final['test_predicted'] = y_test_pred_final['prob'].map(lambda x: 1 if x > 0.4 else 0)


# In[250]:


y_test_pred_final.info()


# In[251]:


y_test_pred_final.test_predicted=y_test_pred_final.test_predicted.astype("int64")
y_test_pred_final.class_los_label=y_test_pred_final.class_los_label.astype("int64")


# In[252]:


confusion = metrics.confusion_matrix(y_test_pred_final['class_los_label'], y_test_pred_final['test_predicted'])
print(confusion)


# In[253]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[254]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test_pred_final['class_los_label'], y_test_pred_final['test_predicted']))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))

#F1 Score
from sklearn.metrics import f1_score
print("F1_Score",f1_score(y_test_pred_final['class_los_label'], y_test_pred_final['test_predicted']))


# In[ ]:





# In[ ]:





# In[ ]:




