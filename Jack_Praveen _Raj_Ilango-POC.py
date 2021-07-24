import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib

import random

xls = pd.ExcelFile('TakenMind-Python-Analytics-Problem-case-study-1-1.xlsx')

df1 = pd.read_excel(xls, 'Existing employees')
df1['left_company']= 1

df2 = pd.read_excel(xls, 'Employees who have left')
df2['left_company']= 0

df3 = pd.concat([df1, df2], axis=0)
del df3['Emp ID']

df3.insert(0, 'New_ID', range(0, 0 + len(df3)))
df3.set_index("New_ID", inplace = True)

df3.dept = pd.Categorical(df3.dept).codes           # 'sales' = 0 'accounting' = 1 'hr' = 2 'technical' = 3 'support' = 4 'management'= 5 'IT' = 6 'product_mng' = 7 'RandD' = 8 'marketing' = 9

df3.salary = pd.Categorical(df3.salary).codes       # low = 1, medium = 2, high = 0



"""""
from sklearn.preprocessing import OneHotEncoder

X=df3.dept

one = OneHotEncoder()
X = one.fit_transform(X).toarray()
print(X)
#print(df3['dept'].value_counts())




#df3.to_csv("clean_dataset.csv")

#print(df3['salary'].value_counts())
"""

plt.figure(figsize=(12,12))
dataplot = sns.heatmap(df3.corr(), linewidths=2, annot=True, annot_kws={'size': 8}, square=True)
plt.savefig('heatmap.png', dpi=400)


"""""

#Seperating 'Left_Company' Column

left_col=df3.iloc[:,6].values
left_data=pd.DataFrame()
i=0
while i<len(left_col):
        if(left_col[i]==1):
            a=[df3.iloc[i,:].values]
            left_temp=pd.DataFrame(a)
            left_data=left_data.append(left_temp)
            i=i+1
        else:
            i=i+1

"""



X = pd.DataFrame(df3)
del X['left_company']
y = df3.left_company.values

#Training and Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#print(X_train, X_test)
#print(y_train, y_test)



# Regression Models

"""""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred_linear = regressor.predict(X_test)

accuracy_l=accuracy_score(y_test,y_pred_linear)				#Accuracy Score: 0.9777777777777777
print("Accuracy Linear:", accuracy_l)
"""""

from sklearn.tree import DecisionTreeClassifier
Classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
Classifier.fit(X_train, y_train)

y_pred_decision = Classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred_decision)

accuracy_d=accuracy_score(y_test,y_pred_decision)				#Accuracy Score: 0.9782222222222222
#print("Accuracy Decision:", accuracy_d)


from sklearn.ensemble import RandomForestClassifier
Classifier1 = RandomForestClassifier(n_estimators = 5, criterion = 'entropy', random_state = 0)
Classifier1.fit(X_train, y_train)
y_pred_RandomForest = Classifier1.predict(X_test)

from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred_RandomForest)

accuracy_r=accuracy_score(y_test,y_pred_RandomForest)				#Accuracy Score: 0.9884444444444445
print("Accuracy of Random Forest:", accuracy_r)



"""""
from sklearn.linear_model import LogisticRegression
classifier2=LogisticRegression(random_state=0)
classifier2.fit(X_train,y_train)
y_pred_LR=classifier2.predict(X_test)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred_LR)

accuracy_d=lr=accuracy_score(y_test,y_pred_LR)				#Accuracy Score: 0.9782222222222222
print("Accuracy LogisticRegression:", accuracy_lr)

"""""

curr_emp = pd.DataFrame(df1)
del curr_emp['left_company']

final_pred_data = pd.DataFrame(curr_emp)
del final_pred_data['Emp ID']

final_pred_data.dept = pd.Categorical(final_pred_data.dept).codes
final_pred_data.salary = pd.Categorical(final_pred_data.salary).codes


final_pred_RandomForest = Classifier1.predict(final_pred_data)



final_pred_df = pd.DataFrame(columns=['Emp Id','satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'dept', 'salary'])


count = 0
print('\nList of Employees Who Might Leave The Company:\n')
for i in final_pred_RandomForest:
    count = count + 1
    if i == 0:
        to_append = curr_emp.iloc[count-1,:].values
        df_length = len(final_pred_df)
        final_pred_df.loc[df_length] = to_append

print(final_pred_df.head(10))

#final_pred_df.to_csv('final_employee_prediction.csv')