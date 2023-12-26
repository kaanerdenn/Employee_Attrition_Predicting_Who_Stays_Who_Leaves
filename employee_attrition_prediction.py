import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Setting display options for pandas DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Loading the dataset
employee_df = pd.read_csv("Human_Resources.csv")
print(employee_df.head())

# Function to provide a general overview of the dataframe
def check_df(dataframe, head=5):
    # Print various insights of the dataframe including shape, datatypes, and quantiles
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# Applying the function to our dataframe
check_df(employee_df, head=2)

# Encoding categorical features
employee_df['Attrition'] = employee_df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['OverTime'] = employee_df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['Over18'] = employee_df['Over18'].apply(lambda x: 1 if x == 'Yes' else 0)

# Dropping unnecessary columns
employee_df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)

# Plotting histograms for each numerical feature
employee_df.hist(bins=10, figsize=(4, 3), color='g')
plt.subplots_adjust(bottom=0.1)
plt.show()

# Correlation heatmap
correlations = employee_df.corr()
sns.heatmap(correlations, annot=True)
plt.show()

# Kernel Density Estimation plots for different features
plt.figure(figsize=[8, 8])
sns.kdeplot(employee_df['YearsWithCurrManager'], label='Years With Current Manager', shade=True)
plt.xlabel('Years With Current Manager')
plt.show()

# Boxplots for Monthly Income by Gender and Job Role
sns.boxplot(x='MonthlyIncome', y='Gender', data=employee_df)
sns.boxplot(x='MonthlyIncome', y='JobRole', data=employee_df)

# One-Hot Encoding categorical variables
X_cat = employee_df[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()
X_cat = pd.DataFrame(X_cat)

# Concatenating One-Hot Encoded categorical data with numerical data
X_numerical = employee_df.select_dtypes(include=[np.number])
X_all = pd.concat([X_cat, X_numerical], axis=1)

# Scaling features
scaler = MinMaxScaler()
X = scaler.fit_transform(X_all)

# Splitting the dataset into training and test sets
y = employee_df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Random Forest Model
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# Performance Evaluation
# Logistic Regression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
confusion_lr = confusion_matrix(y_test, y_pred_lr)
classification_lr = classification_report(y_test, y_pred_lr)

# Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
confusion_rf = confusion_matrix(y_test, y_pred_rf)
classification_rf = classification_report(y_test, y_pred_rf)

# Print the evaluation metrics for each model
print(f"Logistic Regression Accuracy: {accuracy_lr}")
print(f"Random Forest Accuracy: {accuracy_rf}")

print("Logistic Regression Classification Report:")
print(classification_lr)
print("Random Forest Classification Report:")
print(classification_rf)
