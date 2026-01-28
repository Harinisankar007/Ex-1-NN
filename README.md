<H3>NAME : HARINI S</H3>
<H3>REGISTER NO : 212224240049</H3>
<H3>EX.NO : 1</H3>
<H3>DATE : 27.01.2026</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df= pd.read_csv("Churn_Modelling.csv")
print(df)

X=df.iloc[:,:-1].values
print(X)

y=df.iloc[:,-1].values
print(y)

print(df.isnull().sum())
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

print(df.isnull().sum())
y=df.iloc[:,-1].values
print(y)

df.duplicated()
print(df['EstimatedSalary'].describe())

scaler=MinMaxScaler()
df1 = pd.DataFrame(
    scaler.fit_transform(df.select_dtypes(include='number')),
    columns=df.select_dtypes(include='number').columns
)
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

print(X_train)
print(len(X_train))
print(X_test)
print("Lenght of X_test ",len(X_test))

```


## OUTPUT:
## DATASET:
<img width="1187" height="797" alt="image" src="https://github.com/user-attachments/assets/1839c212-94b6-478a-9054-f1af1b771244" />


<img width="577" height="342" alt="image" src="https://github.com/user-attachments/assets/430c10bd-9002-4ab3-8975-3b72fec6f1fd" />


## X VALUES:
<img width="735" height="262" alt="image" src="https://github.com/user-attachments/assets/6ed00be6-4883-4044-a8d7-0eb44bfbf4d9" />


## Y VALUES:
<img width="502" height="126" alt="image" src="https://github.com/user-attachments/assets/5596ad4f-2c44-4a93-850e-b891a29b41b6" />


## NULL VALUES:
<img width="988" height="750" alt="image" src="https://github.com/user-attachments/assets/9048ffd8-90d3-4c92-b4e3-7f43bf389a24" />

<img width="397" height="143" alt="image" src="https://github.com/user-attachments/assets/a71f4f58-5a03-402a-92c8-9153f0d051c9" />


## DUPLICATED VALUES:
<img width="548" height="321" alt="image" src="https://github.com/user-attachments/assets/195e32fc-45cd-410c-84d1-38f54c55aa9b" />


## DESCRIPTION:
<img width="690" height="297" alt="image" src="https://github.com/user-attachments/assets/9197c853-944e-4a4e-8ee9-68324779a5c5" />

## TRAINING DATA:
<img width="1095" height="755" alt="image" src="https://github.com/user-attachments/assets/ecea2033-8b67-4ea4-855f-f8d405ce11bc" />


## TESTING DATA:

<img width="1017" height="518" alt="image" src="https://github.com/user-attachments/assets/37253e9e-80f2-4daf-bd76-093987218035" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


