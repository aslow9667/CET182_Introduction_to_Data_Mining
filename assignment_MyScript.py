import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
import seaborn as sns 
import missingno as msno
# from imblearn.oversampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from kmodes.kprototypes import KPrototypes
from sklearn.model_selection import GridSearchCV

''' Loading Dataset '''
loadedData = pd.read_csv('C:/Users/aslow/Desktop/NTU - Information Management & Analytics/CET182 Introduction to Data Mining/Assignment/Data/adult.csv', \
                         index_col=False,
                 names=['age','workclass','final_weight','education','education-num',\
                        'marital-status','occupation','relationship','race','sex','capital-gain',\
                        'capital-loss','hours-per-week','native-country', 'gross_income'])      # Read CSV dataset   
    
categoricalCols = ['workclass', 'marital-status','occupation',\
                   'relationship','race','sex','native-country', 'gross_income']                 # List of Categorical Columns 

''' Exploratory Data Analysis + Data Cleasning '''
dataInfo = loadedData.describe()                                                              # Get general description of the data, e.g. mean, percentile, max/min
loadedData = loadedData.applymap(lambda x: x.strip() if isinstance(x, str) else x)            # Remove leading space 
loadedData = loadedData.replace('?', np.nan)                                                  # Replace '?' with NaN
msno.matrix(loadedData)                                                                 # Visualise missing value in dataset
# sns.pairplot(loadedData_noMissingValue)                                                 # Pairwise Comparison
loadedData = loadedData.drop(['capital-gain', 'capital-loss','education','final_weight'], axis=1)     # Drop irrelevant columns & duplicated information
loadedData_noMissingValue = loadedData.dropna().reset_index(drop=True)                 # Drop rows with missing value

# =============================================================================
# ''' Data Exploration '''
# loadedData_noMissingValue.hist(figsize=(15,30),layout=(4,2))                           # Plot histgram for all continous variables
# for col in categoricalCols:                                                             # Plot count value of Categorical Columns 
#     df_CatCol = loadedData_noMissingValue[col].value_counts()
#     loadedData_noMissingValue[col].value_counts().plot(kind='bar')
#     plt.show()
# =============================================================================

# =============================================================================
# '''Data Preprocessing: Min-Max Scaler (Sample Weight)'''
# scaler = MinMaxScaler()
# sampleWeight = scaler.fit_transform(loadedData_noMissingValue.pop('final_weight').values.reshape(-1,1))
# =============================================================================

# Scaling Numeric Columns
ss_loadedData_noMissingValue = loadedData_noMissingValue.copy()
ss = StandardScaler()
cols = loadedData_noMissingValue.select_dtypes(exclude='object').columns
ss_loadedData_noMissingValue[cols] = ss.fit_transform(loadedData_noMissingValue[cols])

''''Label Encoding for Categorical Attributes''' 
le = LabelEncoder()
for col in categoricalCols:
    ss_loadedData_noMissingValue[col] = le.fit_transform(loadedData_noMissingValue[col])
    print ('Completed Label Encoder for the Categorical Column: ' + col)

# Getting Categorical Columns
categoricalColPos = [1, 3, 4, 5, 6, 7, 9, 10]

# =============================================================================
# ''' Elbow for number of clusters '''
# cost = []
# for x in range(2,8):
#     kprototype = KPrototypes(n_clusters = x, init = 'Huang', random_state = 0)
#     clusters = kprototype.fit_predict(loadedData_noMissingValue, categorical = categoricalColPos)
#     cost.append(kprototype.cost_)
#     print('Cluster initiation: {}'.format(clusters))
# 
# # Converting the results into a dataframe and plotting them
# df_cost = pd.DataFrame()
# df_cost['clusters'] = range(2,8)
# df_cost['cost'] = cost
# 
# # elbow method for number of clusters
# sns.lineplot(x='clusters', y= 'cost', data=df_cost)
# =============================================================================

''' Clustering - KPrototypes'''
kprototype = KPrototypes(n_clusters=4, random_state=42, init='Huang')
clusters = kprototype.fit_predict(ss_loadedData_noMissingValue, categorical = categoricalColPos)
loadedData_noMissingValue['Cluster'] = kprototype.labels_

features = ['workclass', 'marital-status', 'relationship', 'race', 'sex', 'gross_income']

fig, axes = plt.subplots(3,3, figsize=(10,14))
i=0
for raw in axes:
    for col in raw:
        sns.countplot(x='Cluster', hue='{}'.format(features[i]), data=loadedData_noMissingValue, ax=col)
        col.set_title('{}'.format(features[i]))
        i+=1
        if i==5:
            break
axes[-1, -1].axis('off');



# =============================================================================
# ''''Data Pre-Processing: One Hot Encoding ''' 
# predict_loadedData_noMissingValue = loadedData_noMissingValue.copy
# onehotencoder = OneHotEncoder()
# 
# for col in categoricalCols:
#     X = onehotencoder.fit_transform(predict_loadedData_noMissingValue[col].values.reshape(-1,1)).toarray()
#     dfOneHot = pd.DataFrame(X, columns = onehotencoder.get_feature_names([col])) 
#     predict_loadedData_noMissingValue = pd.concat([predict_loadedData_noMissingValue, dfOneHot], axis=1)
#     predict_loadedData_noMissingValue = predict_loadedData_noMissingValue.drop([col], axis=1) 
#     print ('Completed OneHotEncoded for the Categorical Column: ' + col)
# 
# labelencoder = LabelEncoder()
# predict_loadedData_noMissingValue['gross_income'] = labelencoder.fit_transform(predict_loadedData_noMissingValue['gross_income'])
# =============================================================================

# =============================================================================
''' Random Forest Classifier (Prediction) ''' 
# clf = RandomForestClassifier()
# scaler = StandardScaler()
# 
# y = predict_loadedData_noMissingValue.pop('gross_income')
# feature_name = list(predict_loadedData_noMissingValue.columns)
# x = scaler.fit_transform(predict_loadedData_noMissingValue)
# 
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
# 
# ''' Grid Search '''
# print ('Start Grid Search')
# 
# param_grid = { 
#     'n_estimators': [10, 20, 30, 40],
#     'max_depth' : [4, 8, 12, 16]}
# 
# CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
# CV_rfc.fit(X_train, y_train)
# 
# CV_rfc.best_params_
# 
# ''' Random Forest '''
# clf = RandomForestClassifier(random_state=42, n_estimators= 30, max_depth=16)
# clf.fit(X_train, y_train)
# feature_imp = pd.Series(clf.feature_importances_, index = feature_name).sort_values(ascending = False) # 
# #print (feature_impt)
# #importances = clf.feature_importances_
# 
# y_pred = clf.predict(X_test) # what the model has predicted
# confusion_matrix(y_test, y_pred)
# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# print ('tn:' + str(tn))
# print ('fp:' + str(fp))
# print ('fn:' + str(fn))
# print ('tp:' + str(tp))
# print ('precision:' + str(tp/(tp+fp)))
# print ('recall:' +  str(tp/(tp+fn)))
# =============================================================================

# =============================================================================
''' K folds Validation '''
# '''k_folds = KFold(n_splits = 5)
# scores = cross_val_score(clf, trainData, target, cv = k_folds)
# 
# print("Cross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Number of CV Scores used in Average: ", len(scores))
# =============================================================================



