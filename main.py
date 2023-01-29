import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb
from sklearn import linear_model
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.svm import SVC
import statistics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as ltb
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ================================================================================== #

# Load data
data_train = pd.read_csv('cars-train.csv')
dataframe_train = pd.DataFrame(data_train)

data_test = pd.read_csv('cars-test.csv')
dataframe_test = pd.DataFrame(data_test)

global final_train
global final_test

module_name = ['SVC','LR','NB','KNN','DT','RF','XGBoost','catboost','LGBM']
train_time = []
test_time = []

# ================================================================================== #


def preprocessing_train(d):
    good_data = d['car-info'].str.split(',', n=2, expand=True)
    d = d.drop('car-info', axis=1)
    d = d.drop('car_id', axis=1)

    # Using DataFrame.insert() to add a column
    d.insert(1, "category", good_data[0], True)
    d.insert(2, "car_name", good_data[1], True)
    d.insert(3, "year_production", good_data[2], True)

    # drop na values
    d = d.dropna(axis=0, how='any')

    # fill na values
    # d.fillna(method='ffill', inplace=True)  # ffill backfill bfill pad

    # Convert "category" to string
    d = d.astype({'category': 'string'})
    d['category'] = d['category'].replace("[()[]", "", regex=True)
    d['car_name'] = d['car_name'].replace("[()[]", "", regex=True)
    d['year_production'] = d['year_production'].replace("[()]", "", regex=True)
    d['year_production'] = d['year_production'].replace("]", "", regex=True)

    global final_train
    final_train = d
    final_train = pd.DataFrame(final_train)

# ================================================================================== #


def preprocessing_test(d):
    good_data = d['car-info'].str.split(',', n=2, expand=True)
    d = d.drop('car-info', axis=1)
    d = d.drop('car_id', axis=1)

    # Using DataFrame.insert() to add a column
    d.insert(1, "category", good_data[0], True)
    d.insert(2, "car_name", good_data[1], True)
    d.insert(3, "year_production", good_data[2], True)

    # fill na values
    d.fillna(method='ffill', inplace=True)  # ffill backfill bfill pad

    # Convert "category" to string
    d = d.astype({'category': 'string'})
    d['category'] = d['category'].replace("[()[]", "", regex=True)
    d['car_name'] = d['car_name'].replace("[()[]", "", regex=True)
    d['year_production'] = d['year_production'].replace("[()]", "", regex=True)
    d['year_production'] = d['year_production'].replace("]", "", regex=True)
    global final_test
    final_test = d
    final_test = pd.DataFrame(final_test)

# ================================================================================== #

preprocessing_train(dataframe_train)
preprocessing_test(dataframe_test)

# ================================================================================== #

train_columns = ['condition', 'fuel_type', 'transmission', 'color', 'drive_unit', 'segment', 'category', 'car_name','Price Category']
test_columns = ['condition', 'fuel_type', 'transmission', 'color', 'drive_unit', 'segment', 'category', 'car_name']

# ================================================================================== #

dataframe_train['fuel_type'] = dataframe_train['fuel_type'].str.upper()
dataframe_test['fuel_type'] = dataframe_test['fuel_type'].str.upper()

label_encoder = preprocessing.LabelEncoder()
for i in train_columns:
    final_train[i] = label_encoder.fit_transform(final_train[i])

test_label_encoder = preprocessing.LabelEncoder()
for c in test_columns:
    final_test[c] = label_encoder.fit_transform(final_test[c])

# ================================================================================== #

# # Feature Selection kendall pearson (tooooop spearman)
# cor = final_train.corr(numeric_only=True, method='kendall')
# top_feature = cor.index[cor['Price Category'] > 0.05]
# # # Correlation plot
# # plt.subplots(figsize=(10, 6))
# # top_corr = final_train[top_feature].corr()
# # sns.heatmap(top_corr,  cmap='RdBu', annot=True)#RdBu inferno
# # plt.show()
# top_feature = top_feature.delete(-1)

X = final_train
Y = final_train['Price Category']
X['volume(cm3)'] = X['volume(cm3)'].astype(int)

# ================================================================================== #

# # apply Standard Scaler techniques
# K = X.columns
# ss = StandardScaler()
# X = ss.fit_transform(X)
# X = pd.DataFrame(X, columns=K)


# # apply normalization techniques
# df_max_scaled = X.copy()
# for column in df_max_scaled.columns:
#     df_max_scaled[column] = df_max_scaled[column] / df_max_scaled[column].abs().max()
# X = df_max_scaled

# ================================================================================== #

X=X.drop('Price Category', axis=1)
mutual_info = mutual_info_classif(X, Y)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X.columns
mutual_info.sort_values(ascending=False)

# mutual_info_classif plot
# listtt = list(mutual_info)
# names = ['CON','CAT','C_N','Y_P','M(km)','F_T','V(cm3)','C','T','D_U','S']
# c = ['olive', 'teal', 'darkolivegreen', 'midnightblue', 'green']
# plt.bar(names, listtt,color = c)
# plt.show()

sel_five_cols = SelectKBest(mutual_info_classif, k=11)
sel_five_cols.fit(X, Y)
train = X.columns[sel_five_cols.get_support()]

X_TRAIN = final_train[train]
Y_TRAIN = Y
Z = final_test[train]

# ================================================================================== #

X_train, X_valid, y_train, y_valid = train_test_split(X_TRAIN, Y_TRAIN, random_state = 10 , test_size = 0.2, shuffle=True)

# ================================================================================== #
                                # SVC # will be deleted
# ================================================================================== #

# # 43%(sigmoid) with drop na and no SC
# # Apply Support Vector classification on the selected features
# # rbf sigmoid poly  linear
#
classifier = SVC(kernel='rbf')

# fitting x samples and y classes
S_train_time = time.time()
classifier.fit(X_train, y_train)
E_train_time = time.time()
train_time.append(round(E_train_time - S_train_time,2))

S_test_time = time.time()
prediction = classifier.predict(Z)
E_test_time = time.time()
test_time.append(round((E_test_time - S_test_time)/2,2))
#
# df = pd.DataFrame(prediction, columns=['Price Category'])
#
# for i in range(len(df)):
#     if df.at[i,'Price Category'] == 0:
#         df.at[i, 'Price Category'] = 'cheap'
#     elif df.at[i,'Price Category'] == 1:
#         df.at[i, 'Price Category'] = 'expensive'
#     elif df.at[i, 'Price Category'] == 2:
#         df.at[i, 'Price Category'] = 'moderate'
#     else:
#         df.at[i, 'Price Category'] = 'very expensive'
#
# print(df['Price Category'].value_counts())
# sns.countplot(x ='Price Category', data = df)
# plt.show()
#
# # print(df)
# accuracy = metrics.accuracy_score(y_valid, prediction)
# accuracy_percentage = 100 * accuracy
# print("Accuracy Percentage",accuracy_percentage,"%" )
# # df.to_csv('SVC(sigmoid).csv')

# ================================================================================== #
                                # Logistic Regression # will be deleted
# ================================================================================== #

# # # 80.7% with drop na and SC is working
#
classifier = LogisticRegression()
S_train_time = time.time()
classifier.fit(X_train, y_train)
E_train_time = time.time()
train_time.append(round(E_train_time - S_train_time,2))

S_test_time = time.time()
prediction = classifier.predict(Z)
E_test_time = time.time()
test_time.append(round(E_test_time - S_test_time,2))
#
# df = pd.DataFrame(prediction, columns=['Price Category'])
#
# for i in range(len(df)):
#     if df.at[i,'Price Category'] == 0:
#         df.at[i, 'Price Category'] = 'cheap'
#     elif df.at[i,'Price Category'] == 1:
#         df.at[i, 'Price Category'] = 'expensive'
#     elif df.at[i, 'Price Category'] == 2:
#         df.at[i, 'Price Category'] = 'moderate'
#     else:
#         df.at[i, 'Price Category'] = 'very expensive'
#
# print(df['Price Category'].value_counts())
# sns.countplot(x ='Price Category', data = df)
# plt.show()
#
# accuracy = metrics.accuracy_score(y_valid, prediction)
# accuracy_percentage = 100 * accuracy
# print("Accuracy Percentage",accuracy_percentage,"%" )
# # df.to_csv('SVC(rbf).csv')

# ================================================================================== #
                                # Naive Bayes # will be deleted
# ================================================================================== #

# # 75% with drop na and no SC
#
classifier = GaussianNB()
S_train_time = time.time()
classifier.fit(X_train, y_train)
E_train_time = time.time()
train_time.append(round(E_train_time - S_train_time,2))

S_test_time = time.time()
prediction = classifier.predict(Z)
E_test_time = time.time()
test_time.append(round(E_test_time - S_test_time,2))
#
# df = pd.DataFrame(prediction, columns=['Price Category'])
#
# for i in range(len(df)):
#     if df.at[i,'Price Category'] == 0:
#         df.at[i, 'Price Category'] = 'cheap'
#     elif df.at[i,'Price Category'] == 1:
#         df.at[i, 'Price Category'] = 'expensive'
#     elif df.at[i, 'Price Category'] == 2:
#         df.at[i, 'Price Category'] = 'moderate'
#     else:
#         df.at[i, 'Price Category'] = 'very expensive'
#
# print(df['Price Category'].value_counts())
# sns.countplot(x ='Price Category', data = df)
# plt.show()
#
# accuracy = metrics.accuracy_score(y_valid, prediction)
# accuracy_percentage = 100 * accuracy
# print("Accuracy Percentage",accuracy_percentage,"%" )
# # df.to_csv('new NB(RS=10,drop milage).csv')

# ================================================================================== #
                                # KNN #
# ================================================================================== #

# # 85.1% with drop na and SC is working
#
classifier = KNeighborsClassifier(leaf_size=30,n_neighbors=15) # 100 , 15 n_neighbors=15
S_train_time = time.time()
classifier.fit(X_train, y_train)
E_train_time = time.time()
train_time.append(round(E_train_time - S_train_time,2))

S_test_time = time.time()
prediction = classifier.predict(Z)
E_test_time = time.time()
test_time.append(round(E_test_time - S_test_time,2))
#
# df = pd.DataFrame(prediction, columns=['Price Category'])
#
# for i in range(len(df)):
#     if df.at[i,'Price Category'] == 0:
#         df.at[i, 'Price Category'] = 'cheap'
#     elif df.at[i,'Price Category'] == 1:
#         df.at[i, 'Price Category'] = 'expensive'
#     elif df.at[i, 'Price Category'] == 2:
#         df.at[i, 'Price Category'] = 'moderate'
#     else:
#         df.at[i, 'Price Category'] = 'very expensive'
#
# print(df['Price Category'].value_counts())
# sns.countplot(x ='Price Category', data = df)
# plt.show()
#
# accuracy = metrics.accuracy_score(y_valid, prediction)
# accuracy_percentage = 100 * accuracy
# print("Accuracy Percentage of ",round(accuracy_percentage,3),"%" )
# # df.to_csv('KNN(test(pad)).csv')

# ================================================================================== #
                                # Descision Tree #
# ================================================================================== #

# # 84.5%
# #89.1 mutual_info
#
#
S_train_time = time.time()
dtree_model = DecisionTreeClassifier(max_depth=None).fit(X_train, y_train)
E_train_time = time.time()
train_time.append(round(E_train_time - S_train_time,2))

S_test_time = time.time()
dtree_predictions = dtree_model.predict(Z)
E_test_time = time.time()
test_time.append(round(E_test_time - S_test_time,2))
#
# df = pd.DataFrame(dtree_predictions, columns=['Price Category'])
#
#
# for i in range(len(df)):
#     if df.at[i,'Price Category'] == 0:
#         df.at[i, 'Price Category'] = 'cheap'
#     elif df.at[i,'Price Category'] == 1:
#         df.at[i, 'Price Category'] = 'expensive'
#     elif df.at[i, 'Price Category'] == 2:
#         df.at[i, 'Price Category'] = 'moderate'
#     else:
#         df.at[i, 'Price Category'] = 'very expensive'
#
# print(df['Price Category'].value_counts())
# sns.countplot(x ='Price Category', data = df)
# plt.show()
#
# accuracy = metrics.accuracy_score(y_valid, dtree_predictions)
# accuracy_percentage = 100 * accuracy
# print("Accuracy Percentage = ",accuracy_percentage,"%" )
# # df.to_csv('DT(K=8,RS=30,3).csv')

# ================================================================================== #
                                # Random Forest #
# ================================================================================== #

# # 85.6% with drop na and SC is working
# # 84.92
# # 89.7 mutual_info
#
clf_4 = RandomForestClassifier()#criterion='entropy',max_features='log2'
S_train_time = time.time()
clf_4.fit(X_train, y_train)
E_train_time = time.time()
train_time.append(round(E_train_time - S_train_time,2))

S_test_time = time.time()
pred_y_4 = clf_4.predict(Z)
E_test_time = time.time()
test_time.append(round(E_test_time - S_test_time,2))
#
# df = pd.DataFrame(pred_y_4, columns=['Price Category'])
#
# for i in range(len(df)):
#     if df.at[i,'Price Category'] == 0:
#         df.at[i, 'Price Category'] = 'cheap'
#     elif df.at[i,'Price Category'] == 1:
#         df.at[i, 'Price Category'] = 'expensive'
#     elif df.at[i, 'Price Category'] == 2:
#         df.at[i, 'Price Category'] = 'moderate'
#     else:
#         df.at[i, 'Price Category'] = 'very expensive'
#
# print(df['Price Category'].value_counts())
# sns.countplot(x ='Price Category', data = df)
# plt.show()
#
# # print(df)
# accuracy = metrics.accuracy_score(y_valid, pred_y_4)
# accuracy_percentage = 100 * accuracy
# print("Accuracy Percentage = ",accuracy_percentage,"%" )
# # df.to_csv('RF(K=8,RS=30).csv')

# ================================================================================== #
                                # XGBoost #
# ================================================================================== #

# #84.9%
# #89.1
Z['year_production']= Z['year_production'].astype(str).astype(int)
Z['category']= Z['category'].astype(str).astype(int)
X_train['year_production']= X_train['year_production'].astype(str).astype(int)
X_train['category']= X_train['category'].astype(str).astype(int)
X_valid['year_production']= X_valid['year_production'].astype(str).astype(int)
X_valid['category']= X_valid['category'].astype(str).astype(int)
# # approx gpu_hist hist
clf= xgb.XGBClassifier(tree_method="gpu_hist",enable_categorical=True)
S_train_time = time.time()
clf.fit(X_train, y_train)
E_train_time = time.time()
train_time.append(round(E_train_time - S_train_time,2))

S_test_time = time.time()
prediction = clf.predict(Z)
E_test_time = time.time()
test_time.append(round(E_test_time - S_test_time,2))
#
# df = pd.DataFrame(prediction, columns=['Price Category'])
#
# for i in range(len(df)):
#     if df.at[i,'Price Category'] == 0:
#         df.at[i, 'Price Category'] = 'cheap'
#     elif df.at[i,'Price Category'] == 1:
#         df.at[i, 'Price Category'] = 'expensive'
#     elif df.at[i, 'Price Category'] == 2:
#         df.at[i, 'Price Category'] = 'moderate'
#     else:
#         df.at[i, 'Price Category'] = 'very expensive'
#
# print(df['Price Category'].value_counts())
# sns.countplot(x ='Price Category', data = df)
# plt.show()
#
# accuracy = metrics.accuracy_score(y_valid, prediction)
# accuracy_percentage = 100 * accuracy
# print("Accuracy Percentage = ",accuracy_percentage,"%" )
# # df.to_csv('XGB(K=8,RS=30).csv')

# ================================================================================== #
                                # catboost #
# ================================================================================== #

# # 85.33%
# # 89.2
# # CBC = CatBoostClassifier()
# # parameters = {'depth': [4, 5, 6, 7, 8, 9, 10],
# #               'learning_rate': [0.01, 0.02, 0.03, 0.04],
# #               'iterations': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# #               }
# # Grid_CBC = GridSearchCV(estimator=CBC, param_grid=parameters, cv=2, n_jobs=-1)
# # Grid_CBC.fit(X_train, y_train)
# #
# # print(" Results from Grid Search ")
# # print("\n The best estimator across ALL searched params:\n", Grid_CBC.best_estimator_)
# # print("\n The best score across ALL searched params:\n", Grid_CBC.best_score_)
# # print("\n The best parameters across ALL searched params:\n", Grid_CBC.best_params_)
#
model = CatBoostClassifier() #iterations=1000,task_type="GPU",devices='0:1'
# # The best parameters across ALL searched params:{'depth': 10, 'iterations': 100, 'learning_rate': 0.04}
# S_train_time = time.time()
model.fit(X_train, y_train,verbose=False)
E_train_time = time.time()
train_time.append(round(E_train_time - S_train_time,2))

S_test_time = time.time()
prediction = model.predict(Z)
E_test_time = time.time()
test_time.append(round(E_test_time - S_test_time,2))
#
# df = pd.DataFrame(prediction, columns=['Price Category'])
#
# for i in range(len(df)):
#     if df.at[i,'Price Category'] == 0:
#         df.at[i, 'Price Category'] = 'cheap'
#     elif df.at[i,'Price Category'] == 1:
#         df.at[i, 'Price Category'] = 'expensive'
#     elif df.at[i, 'Price Category'] == 2:
#         df.at[i, 'Price Category'] = 'moderate'
#     else:
#         df.at[i, 'Price Category'] = 'very expensive'
#
# print(df['Price Category'].value_counts())
# sns.countplot(x ='Price Category', data = df)
# plt.show()
#
# accuracy = metrics.accuracy_score(y_valid, prediction)
# accuracy_percentage = 100 * accuracy
# print("Accuracy Percentage = ",accuracy_percentage,"%" )
# # df.to_csv('new catboost(RS=10,dropNA).csv')

# ================================================================================== #
                                # LGBMClassifier #
# ================================================================================== #

# # 89.2
Z['year_production']= Z['year_production'].astype(str).astype(int)
Z['category']= Z['category'].astype(str).astype(int)
X_train['year_production']= X_train['year_production'].astype(str).astype(int)
X_train['category']= X_train['category'].astype(str).astype(int)
X_valid['year_production']= X_valid['year_production'].astype(str).astype(int)
X_valid['category']= X_valid['category'].astype(str).astype(int)
#
model = ltb.LGBMClassifier()
S_train_time = time.time()
model.fit(X_train, y_train)
E_train_time = time.time()
train_time.append(round(E_train_time - S_train_time,2))

S_test_time = time.time()
prediction = model.predict(Z)
E_test_time = time.time()
test_time.append(round(E_test_time - S_test_time,2))

# df = pd.DataFrame(prediction, columns=['Price Category'])
#
# for i in range(len(df)):
#     if df.at[i,'Price Category'] == 0:
#         df.at[i, 'Price Category'] = 'cheap'
#     elif df.at[i,'Price Category'] == 1:
#         df.at[i, 'Price Category'] = 'expensive'
#     elif df.at[i, 'Price Category'] == 2:
#         df.at[i, 'Price Category'] = 'moderate'
#     else:
#         df.at[i, 'Price Category'] = 'very expensive'
#
# print(df['Price Category'].value_counts())
# sns.countplot(x ='Price Category', data = df)
# plt.show()
#
# accuracy = metrics.accuracy_score(y_valid, prediction)
# accuracy_percentage = 100 * accuracy
# print("Accuracy Percentage = ",accuracy_percentage,"%" )

# df.to_csv('LGBMClassifier.csv')

# ================================================================================== #
                                # bar plot #
# ================================================================================== #

x = np.arange(len(module_name))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, train_time, width, label='train_time')
rects2 = ax.bar(x + width/2, test_time, width, label='test_time')

ax.set_ylabel('Rating')
ax.set_title('Scores of Train and predict')
ax.set_xticks(x, module_name)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
fig.tight_layout()
plt.show()

