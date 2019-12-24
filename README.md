# BUILDING A DEFUALT DETECTION MODEL

---

By Reon Ho, Lam Cheng Jun, Janson Chew, and Bryan Koh



## Table of Contents
1. Problem Description (Brief Write Up)
2. Exploratory Data Analysis (EDA)
3. Data Pre-processing
4. Model Selection
5. Evaluation
6. Discussion and Possible Improvements

## 1. Problem Description

The goal of this project is to predict a binary target feature (default or not) valued 0 (= not default) or 1 (= default). This project will cover the entire data science pipeline, from data analysis to model evaluation. We will be trying several models to predict default status, and choosing the most appropriate one at the end. 

The data set we will be working on contains payment information of 30,000 credit card holders obtained from a bank in Taiwan, and each data sample is described by 23 feature attributes and the binary target feature (default or not).

The 23 explanatory attributes and their explanations (from the data provider) are as follows:

### X1 - X5: Indivual attributes of customer

X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit. 

X2: Gender (1 = male; 2 = female). 

X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others). 

X4: Marital status (1 = married; 2 = single; 3 = others). 

X5: Age (year). 

### X6 - X11: Repayment history from April to Septemeber 2005
The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months, . . . 8 = payment delay for eight months; 9 = payment delay for nine months and above.


X6 = the repayment status in September, 2005

X7 = the repayment status in August, 2005

X8 = the repayment status in July, 2005

X9 = the repayment status in June, 2005

X10 = the repayment status in May, 2005

X11 = the repayment status in April, 2005. 

### X12 - X17: Amount of bill statement (NT dollar) from April to September 2005

X12 = amount of bill statement in September, 2005; 

X13 = amount of bill statement in August, 2005

. . .

X17 = amount of bill statement in April, 2005. 

### X18 - X23: Amount of previous payment (NT dollar)
X18 = amount paid in September, 2005

X19 = amount paid in August, 2005

. . .

X23 = amount paid in April, 2005. 


## EDA

In this section we will explore the data set, its shape and its features to get an idea of the data.

### Importing packages and the dataset


```python
import pandas as pd
```


```python
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
import numpy as np
```


```python
url = 'https://raw.githubusercontent.com/reonho/bt2101disrudy/master/card.csv'
df = pd.read_csv(url,  header = 1, index_col = 0)
# Dataset is now stored in a Pandas Dataframe
```


```python
#rename the target variable to "Y" for convenience
df["Y"] = df["default payment next month"] 
df = df.drop("default payment next month", axis = 1)
df0 = df #backup of df
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>Y</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>20000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>689</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>120000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272</td>
      <td>3455</td>
      <td>3261</td>
      <td>0</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>0</td>
      <td>2000</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>90000</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331</td>
      <td>14948</td>
      <td>15549</td>
      <td>1518</td>
      <td>1500</td>
      <td>1000</td>
      <td>1000</td>
      <td>1000</td>
      <td>5000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>50000</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314</td>
      <td>28959</td>
      <td>29547</td>
      <td>2000</td>
      <td>2019</td>
      <td>1200</td>
      <td>1100</td>
      <td>1069</td>
      <td>1000</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>50000</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>20940</td>
      <td>19146</td>
      <td>19131</td>
      <td>2000</td>
      <td>36681</td>
      <td>10000</td>
      <td>9000</td>
      <td>689</td>
      <td>679</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
size = df.shape
print("Data has {} Columns and {} Rows".format(size[1], size[0]))
```

    Data has 24 Columns and 30000 Rows
    


```python
#check for null values
df.isnull().any().sum() 
```




    0



From the above analyses, we observe that:
1. The data indeed has 30000 rows and 24 columns
2. There are no null values

We will now explore the features more in depth.

### Exploring the features

**1) Exploring target attribute:**



```python
All = df.shape[0]
default = df[df['Y'] == 1]
nondefault = df[df['Y'] == 0]

x = len(default)/All
y = len(nondefault)/All

print('defaults :',x*100,'%')
print('non defaults :',y*100,'%')

# plotting target attribute against frequency
labels = ['non default','default']
classes = pd.value_counts(df['Y'], sort = True)
classes.plot(kind = 'bar', rot=0)
plt.title("Target attribute distribution")
plt.xticks(range(2), labels)
plt.xlabel("Class")
plt.ylabel("Frequency")
```

    defaults : 22.12 %
    non defaults : 77.88000000000001 %
    




    Text(0, 0.5, 'Frequency')




![png](defaults_files/defaults_12_2.png)


**2) Exploring categorical attributes**

Categorical attributes are:
- Sex
- Education
- Marriage


```python
print(df["SEX"].value_counts().apply(lambda r: r/All*100))
print("--------------------------------------------------------")
print(df["EDUCATION"].value_counts().apply(lambda r: r/All*100))
print("--------------------------------------------------------")
print(df["MARRIAGE"].value_counts().apply(lambda r: r/All*100))
```

    2    60.373333
    1    39.626667
    Name: SEX, dtype: float64
    --------------------------------------------------------
    2    46.766667
    1    35.283333
    3    16.390000
    5     0.933333
    4     0.410000
    6     0.170000
    0     0.046667
    Name: EDUCATION, dtype: float64
    --------------------------------------------------------
    2    53.213333
    1    45.530000
    3     1.076667
    0     0.180000
    Name: MARRIAGE, dtype: float64
    

**Findings**

- Categorical variable SEX does not seem to have any missing/extra groups, and it is separated into Male = 1 and Female = 2
- Categorical variable MARRIAGE seems to have unknown group = 0, which could be assumed to be missing data, with other groups being Married = 1, Single = 2, Others = 3
- Categorical variable EDUCATION seems to have unknown group = 0,5,6, with other groups being graduate school = 1, university = 2, high school = 3, others = 4 


```python
#proportion of target attribute (for reference)
print('Total target attributes:')
print('non defaults :',y*100,'%')
print('defaults :',x*100,'%')
print("--------------------------------------------------------")
#analysing default payment with Sex
sex_target = pd.crosstab(df["Y"], df["SEX"]).apply(lambda r: r/r.sum()*100).rename(columns = {1: "Male", 2: "Female"}, index = {0: "non defaults", 1: "defaults"})
print(sex_target)
print("--------------------------------------------------------")
#analysing default payment with education
education_target = pd.crosstab(df["Y"], df["EDUCATION"]).apply(lambda r: r/r.sum()*100).rename(index = {0: "non defaults", 1: "defaults"})
print(education_target)
print("--------------------------------------------------------")
#analysing default payment with marriage
marriage_target = pd.crosstab(df["Y"], df["MARRIAGE"]).apply(lambda r: r/r.sum()*100).rename(columns = {0: "unknown",1: "married", 2: "single", 3: "others"},index = {0: "non defaults", 1: "defaults"})
print(marriage_target)
```

    Total target attributes:
    non defaults : 77.88000000000001 %
    defaults : 22.12 %
    --------------------------------------------------------
    SEX                Male     Female
    Y                                 
    non defaults  75.832773  79.223719
    defaults      24.167227  20.776281
    --------------------------------------------------------
    EDUCATION         0          1          2          3          4          5  \
    Y                                                                            
    non defaults  100.0  80.765234  76.265146  74.842384  94.308943  93.571429   
    defaults        0.0  19.234766  23.734854  25.157616   5.691057   6.428571   
    
    EDUCATION             6  
    Y                        
    non defaults  84.313725  
    defaults      15.686275  
    --------------------------------------------------------
    MARRIAGE        unknown    married     single     others
    Y                                                       
    non defaults  90.740741  76.528296  79.071661  73.993808
    defaults       9.259259  23.471704  20.928339  26.006192
    

**Conclusion**

From the analyses above we conclude that

1. The categorical data is noisy - EDUCATION and MARRIAGE contains unexplained/anomalous data.


**3) Analysis of Numerical Attributes**

The numerical attributes are:
   





```python
#printing numerical attributes
pd.DataFrame(df.drop(['SEX', 'EDUCATION', 'MARRIAGE','Y'], axis = 1).columns).transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>LIMIT_BAL</td>
      <td>AGE</td>
      <td>PAY_0</td>
      <td>PAY_2</td>
      <td>PAY_3</td>
      <td>PAY_4</td>
      <td>PAY_5</td>
      <td>PAY_6</td>
      <td>BILL_AMT1</td>
      <td>BILL_AMT2</td>
      <td>BILL_AMT3</td>
      <td>BILL_AMT4</td>
      <td>BILL_AMT5</td>
      <td>BILL_AMT6</td>
      <td>PAY_AMT1</td>
      <td>PAY_AMT2</td>
      <td>PAY_AMT3</td>
      <td>PAY_AMT4</td>
      <td>PAY_AMT5</td>
      <td>PAY_AMT6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(['SEX', 'EDUCATION', 'MARRIAGE','Y'], axis=1).describe().transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LIMIT_BAL</td>
      <td>30000.0</td>
      <td>167484.322667</td>
      <td>129747.661567</td>
      <td>10000.0</td>
      <td>50000.00</td>
      <td>140000.0</td>
      <td>240000.00</td>
      <td>1000000.0</td>
    </tr>
    <tr>
      <td>AGE</td>
      <td>30000.0</td>
      <td>35.485500</td>
      <td>9.217904</td>
      <td>21.0</td>
      <td>28.00</td>
      <td>34.0</td>
      <td>41.00</td>
      <td>79.0</td>
    </tr>
    <tr>
      <td>PAY_0</td>
      <td>30000.0</td>
      <td>-0.016700</td>
      <td>1.123802</td>
      <td>-2.0</td>
      <td>-1.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>PAY_2</td>
      <td>30000.0</td>
      <td>-0.133767</td>
      <td>1.197186</td>
      <td>-2.0</td>
      <td>-1.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>PAY_3</td>
      <td>30000.0</td>
      <td>-0.166200</td>
      <td>1.196868</td>
      <td>-2.0</td>
      <td>-1.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>PAY_4</td>
      <td>30000.0</td>
      <td>-0.220667</td>
      <td>1.169139</td>
      <td>-2.0</td>
      <td>-1.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>PAY_5</td>
      <td>30000.0</td>
      <td>-0.266200</td>
      <td>1.133187</td>
      <td>-2.0</td>
      <td>-1.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>PAY_6</td>
      <td>30000.0</td>
      <td>-0.291100</td>
      <td>1.149988</td>
      <td>-2.0</td>
      <td>-1.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <td>BILL_AMT1</td>
      <td>30000.0</td>
      <td>51223.330900</td>
      <td>73635.860576</td>
      <td>-165580.0</td>
      <td>3558.75</td>
      <td>22381.5</td>
      <td>67091.00</td>
      <td>964511.0</td>
    </tr>
    <tr>
      <td>BILL_AMT2</td>
      <td>30000.0</td>
      <td>49179.075167</td>
      <td>71173.768783</td>
      <td>-69777.0</td>
      <td>2984.75</td>
      <td>21200.0</td>
      <td>64006.25</td>
      <td>983931.0</td>
    </tr>
    <tr>
      <td>BILL_AMT3</td>
      <td>30000.0</td>
      <td>47013.154800</td>
      <td>69349.387427</td>
      <td>-157264.0</td>
      <td>2666.25</td>
      <td>20088.5</td>
      <td>60164.75</td>
      <td>1664089.0</td>
    </tr>
    <tr>
      <td>BILL_AMT4</td>
      <td>30000.0</td>
      <td>43262.948967</td>
      <td>64332.856134</td>
      <td>-170000.0</td>
      <td>2326.75</td>
      <td>19052.0</td>
      <td>54506.00</td>
      <td>891586.0</td>
    </tr>
    <tr>
      <td>BILL_AMT5</td>
      <td>30000.0</td>
      <td>40311.400967</td>
      <td>60797.155770</td>
      <td>-81334.0</td>
      <td>1763.00</td>
      <td>18104.5</td>
      <td>50190.50</td>
      <td>927171.0</td>
    </tr>
    <tr>
      <td>BILL_AMT6</td>
      <td>30000.0</td>
      <td>38871.760400</td>
      <td>59554.107537</td>
      <td>-339603.0</td>
      <td>1256.00</td>
      <td>17071.0</td>
      <td>49198.25</td>
      <td>961664.0</td>
    </tr>
    <tr>
      <td>PAY_AMT1</td>
      <td>30000.0</td>
      <td>5663.580500</td>
      <td>16563.280354</td>
      <td>0.0</td>
      <td>1000.00</td>
      <td>2100.0</td>
      <td>5006.00</td>
      <td>873552.0</td>
    </tr>
    <tr>
      <td>PAY_AMT2</td>
      <td>30000.0</td>
      <td>5921.163500</td>
      <td>23040.870402</td>
      <td>0.0</td>
      <td>833.00</td>
      <td>2009.0</td>
      <td>5000.00</td>
      <td>1684259.0</td>
    </tr>
    <tr>
      <td>PAY_AMT3</td>
      <td>30000.0</td>
      <td>5225.681500</td>
      <td>17606.961470</td>
      <td>0.0</td>
      <td>390.00</td>
      <td>1800.0</td>
      <td>4505.00</td>
      <td>896040.0</td>
    </tr>
    <tr>
      <td>PAY_AMT4</td>
      <td>30000.0</td>
      <td>4826.076867</td>
      <td>15666.159744</td>
      <td>0.0</td>
      <td>296.00</td>
      <td>1500.0</td>
      <td>4013.25</td>
      <td>621000.0</td>
    </tr>
    <tr>
      <td>PAY_AMT5</td>
      <td>30000.0</td>
      <td>4799.387633</td>
      <td>15278.305679</td>
      <td>0.0</td>
      <td>252.50</td>
      <td>1500.0</td>
      <td>4031.50</td>
      <td>426529.0</td>
    </tr>
    <tr>
      <td>PAY_AMT6</td>
      <td>30000.0</td>
      <td>5215.502567</td>
      <td>17777.465775</td>
      <td>0.0</td>
      <td>117.75</td>
      <td>1500.0</td>
      <td>4000.00</td>
      <td>528666.0</td>
    </tr>
  </tbody>
</table>
</div>



**Analysis of PAY_0 to PAY_6**

We observe that the minimum value of PAY_0 to PAY_6 is -2. The dataset's author has explained these factors (PAY_0 to PAY_6) as the number of months of payment delay, that is, 1= payment delay of one month; 2= payment delay of two months and so on. 

However, the presence of -2, -1 in these columns indicates that
1. There is anomalous data, OR 
2. The numbers do not strictly correspond to the number of months of payment delay. 

This means we must conduct some data transformation.

According to the datasets' author, the numeric value in these attributes shows the past history of a credit card holder, where -2 means: No consumption of credit card, -1 means that holder paid the full balance, and 0 means the use of revolving credit.




```python
def draw_histograms(df, variables, n_rows, n_cols, n_bins):
    fig=plt.figure()
    for i, var_name in enumerate(variables):
        ax=fig.add_subplot(n_rows,n_cols,i+1)
        df[var_name].hist(bins=n_bins,ax=ax)
        ax.set_title(var_name)
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

PAY = df[['PAY_0','PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]
BILLAMT = df[['BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']]
PAYAMT = df[['PAY_AMT1','PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]

draw_histograms(PAY, PAY.columns, 2, 3, 10)
draw_histograms(BILLAMT, BILLAMT.columns, 2, 3, 10)
draw_histograms(PAYAMT, PAYAMT.columns, 2, 3, 10)
```


![png](defaults_files/defaults_22_0.png)



![png](defaults_files/defaults_22_1.png)



![png](defaults_files/defaults_22_2.png)


We observe that the "repayment status" attributes are the most highly correlated with the target variable and we would expect them to be more significant in predicting credit default. In fact the later the status (pay_0 is later than pay_6), the more correlated it is.

Now that we have an idea of the features, we will move on to feature selection and data preparation.

## Data Preprocessing

It was previously mentioned that our data had a bit of noise, so we will clean up the data in this part. Additionally, we will conduct some feature selection.
1. Removing Noise - Inconsistencies
2. Dealing with negative values of PAY_0 to PAY_6
3. Outliers
4. One Hot Encoding
5. Train Test Split
6. Feature selection


### Removing Noise
First, we found in our data exploration that education has unknown groups 0, 5 and 6. These will be dealt with using the identification method. 0 will be assumed to be missing data and identified. Groups 5 and 6 will be subsumed by Education = Others, with value 4


```python
df['EDUCATION'].replace([5,6], 4, regex=True, inplace=True)
df["EDUCATION"].unique()
```




    array([2, 1, 3, 4, 0], dtype=int64)



Similarly, for Marriage, we will use the identification method to deal with missing data. So 0 will be treated as a new category, "Missing"

### Separating negative and positive values for PAY_0 to PAY_6

Second, we are going to extract the negative values of PAY_0 to PAY_6 as another categorical feature. This way, PAY_0 to PAY_6 can be thought of purely as the months of delay of payments.

The negative values will form a categorical variable. e.g. negative values of PAY_0 will form the categorical variable S_0.


```python
for i in range(0,7):
    try:
        df["S_" + str(i)] = [x  if x < 1 else 1 for x in df["PAY_" + str(i)]]
    except:
        pass
```


```python
print('Dummy variables for negative values')
df[["S_0", "S_2", "S_3", "S_4", "S_5", "S_6"]].head()
```

    Dummy variables for negative values
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>S_0</th>
      <th>S_2</th>
      <th>S_3</th>
      <th>S_4</th>
      <th>S_5</th>
      <th>S_6</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>-2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>-1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#attributes representing positive values
for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
    df[col].replace([0,-1,-2], 0, regex=True, inplace=True)
```

### Outliers
Next, we would like to remove outliers from the continuous variables. Assuming that all the data points are normally distributed, we will consider a point an outlier if it falls outside the 99% interval of a distribution. (Critical value = 2.58) 


```python
from scipy import stats
#we are only concerned with the ordinal data
o = pd.DataFrame(df.drop(['Y','EDUCATION', 'MARRIAGE', "SEX","S_0", "S_2", "S_3", "S_4", "S_5", "S_6","PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"], axis=1))
#rows where the absolute z score of all columns are less than 2.58 (critical value)
rows = (np.abs(stats.zscore(o)) < 2.58).all(axis=1)
df = df[rows]
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>...</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>Y</th>
      <th>S_0</th>
      <th>S_2</th>
      <th>S_3</th>
      <th>S_4</th>
      <th>S_5</th>
      <th>S_6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>26245.000000</td>
      <td>26245.000000</td>
      <td>26245.000000</td>
      <td>26245.000000</td>
      <td>26245.000000</td>
      <td>26245.000000</td>
      <td>26245.000000</td>
      <td>26245.000000</td>
      <td>26245.000000</td>
      <td>26245.000000</td>
      <td>...</td>
      <td>26245.000000</td>
      <td>26245.000000</td>
      <td>26245.000000</td>
      <td>26245.000000</td>
      <td>26245.000000</td>
      <td>26245.000000</td>
      <td>26245.000000</td>
      <td>26245.000000</td>
      <td>26245.000000</td>
      <td>26245.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>149324.899981</td>
      <td>1.608954</td>
      <td>1.850753</td>
      <td>1.558773</td>
      <td>35.006592</td>
      <td>0.372109</td>
      <td>0.337321</td>
      <td>0.324633</td>
      <td>0.278224</td>
      <td>0.238750</td>
      <td>...</td>
      <td>2787.425071</td>
      <td>2778.830673</td>
      <td>2822.285007</td>
      <td>0.230177</td>
      <td>-0.133587</td>
      <td>-0.300438</td>
      <td>-0.327300</td>
      <td>-0.364412</td>
      <td>-0.395999</td>
      <td>-0.428158</td>
    </tr>
    <tr>
      <td>std</td>
      <td>116558.616530</td>
      <td>0.487994</td>
      <td>0.738175</td>
      <td>0.522639</td>
      <td>8.832028</td>
      <td>0.765730</td>
      <td>0.814878</td>
      <td>0.811491</td>
      <td>0.786314</td>
      <td>0.743923</td>
      <td>...</td>
      <td>4835.081906</td>
      <td>4751.263287</td>
      <td>5271.198100</td>
      <td>0.420954</td>
      <td>0.879876</td>
      <td>0.883472</td>
      <td>0.895264</td>
      <td>0.886115</td>
      <td>0.877789</td>
      <td>0.900723</td>
    </tr>
    <tr>
      <td>min</td>
      <td>10000.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
      <td>-2.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>50000.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>150.000000</td>
      <td>82.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>120000.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>34.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>1200.000000</td>
      <td>1218.000000</td>
      <td>1143.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>210000.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>41.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>3118.000000</td>
      <td>3140.000000</td>
      <td>3069.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>500000.000000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>59.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>8.000000</td>
      <td>...</td>
      <td>45171.000000</td>
      <td>44197.000000</td>
      <td>51000.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 30 columns</p>
</div>



### Feature Scaling
The models used subsequently may have difficulty converging before the maximum number of iterations allowed
is reached if the data is not normalized. Additionaly, Multi-layer Perceptron is sensitive to feature scaling, so we will use StandardScaler for standardization. We only want to scale the numerical factors.


```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
cols = df.drop(['Y','EDUCATION', 'MARRIAGE', "SEX","S_0", "S_2", "S_3", "S_4", "S_5", "S_6","PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"], axis =1)
df1 = df.copy()
df1[cols.columns] = scaler.fit_transform(cols)
df = df1
```


```python
df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>...</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>Y</th>
      <th>S_0</th>
      <th>S_2</th>
      <th>S_3</th>
      <th>S_4</th>
      <th>S_5</th>
      <th>S_6</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.020408</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0.078947</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-2</td>
      <td>-2</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.224490</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0.131579</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.022138</td>
      <td>0.000000</td>
      <td>0.039216</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.163265</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0.342105</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.022138</td>
      <td>0.022626</td>
      <td>0.098039</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.081633</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0.421053</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.024352</td>
      <td>0.024187</td>
      <td>0.019608</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.081633</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0.947368</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0.199243</td>
      <td>0.015589</td>
      <td>0.013314</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>



### One-Hot Encoding for Categorical attributes

In some models, categorical variables which are encoded numerically will be erroneously treated as ordinal data. To understand why this is a problem, consider the "Education" column for our dataset.

A logistic regression model, for example, will assume that the difference in odds of default between education = 1 and education = 2 is the same as the difference between education = 2 and 3. This is wrong because the difference in odds between a graduate degree and university (1 and 2) is likely to be different from that between univeristy education and high school education (2 and 3).

One hot encoding will allow our models to treat these columns explicitly as categorical features.

The following categorical columns will be one-hot encoded

1. EDUCATION
2. MARRIAGE
3. S0 - S6



```python
from sklearn.preprocessing import OneHotEncoder
```


```python
onenc = OneHotEncoder(categories='auto')
```


```python
#one hot encoding for EDUCATION and MARRIAGE
onehot = pd.DataFrame(onenc.fit_transform(df[['EDUCATION', 'MARRIAGE']]).toarray())
onehot.columns= names = ["MISSING-EDU","GRAD","UNI","HS","OTHER-EDU","MISSING-MS","MARRIED","SINGLE","OTHER-MS"]
#drop one of each category to prevent dummy variable trap
onehot = onehot.drop(["OTHER-EDU", "OTHER-MS"], axis = 1)
onehot.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MISSING-EDU</th>
      <th>GRAD</th>
      <th>UNI</th>
      <th>HS</th>
      <th>MISSING-MS</th>
      <th>MARRIED</th>
      <th>SINGLE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#one hot encoding for S_0 to S_6
onehot_PAY = pd.DataFrame(onenc.fit_transform(df[['S_0', 'S_2', 'S_3', 'S_4', 'S_5', 'S_6']]).toarray())
onehot_PAY.columns= onenc.fit(df[["S_0", "S_2", "S_3", "S_4", "S_5", "S_6"]]).get_feature_names()
#drop one of each category to prevent dummy variable trap
#onehot = onehot.drop(["OTHER-EDU", "OTHER_MS"], axis = 1)
names = []
for X in range(0,7):
    if X == 1:
        continue
    names.append("PAY_"+str(X)+"_No_Transactions")
    names.append("PAY_"+str(X)+"_Pay_Duly")
    names.append("PAY_"+str(X)+"_Revolving_Credit")
    try:
        onehot_PAY = onehot_PAY.drop("x" + str(X) +"_1", axis =1)
    except:
        onehot_PAY = onehot_PAY.drop("x1_1", axis =1)
onehot_PAY.columns = names
onehot_PAY.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PAY_0_No_Transactions</th>
      <th>PAY_0_Pay_Duly</th>
      <th>PAY_0_Revolving_Credit</th>
      <th>PAY_2_No_Transactions</th>
      <th>PAY_2_Pay_Duly</th>
      <th>PAY_2_Revolving_Credit</th>
      <th>PAY_3_No_Transactions</th>
      <th>PAY_3_Pay_Duly</th>
      <th>PAY_3_Revolving_Credit</th>
      <th>PAY_4_No_Transactions</th>
      <th>PAY_4_Pay_Duly</th>
      <th>PAY_4_Revolving_Credit</th>
      <th>PAY_5_No_Transactions</th>
      <th>PAY_5_Pay_Duly</th>
      <th>PAY_5_Revolving_Credit</th>
      <th>PAY_6_No_Transactions</th>
      <th>PAY_6_Pay_Duly</th>
      <th>PAY_6_Revolving_Credit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1 = df.drop(['EDUCATION', 'MARRIAGE','S_0', 'S_2', 'S_3', 'S_4', 'S_5', 'S_6'], axis = 1)
df1 = pd.concat([df1.reset_index(drop=True), onehot], axis=1)
df1 = pd.concat([df1.reset_index(drop=True), onehot_PAY], axis=1)
df1.columns
```




    Index(['LIMIT_BAL', 'SEX', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5',
           'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
           'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
           'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'Y', 'MISSING-EDU', 'GRAD', 'UNI',
           'HS', 'MISSING-MS', 'MARRIED', 'SINGLE', 'PAY_0_No_Transactions',
           'PAY_0_Pay_Duly', 'PAY_0_Revolving_Credit', 'PAY_2_No_Transactions',
           'PAY_2_Pay_Duly', 'PAY_2_Revolving_Credit', 'PAY_3_No_Transactions',
           'PAY_3_Pay_Duly', 'PAY_3_Revolving_Credit', 'PAY_4_No_Transactions',
           'PAY_4_Pay_Duly', 'PAY_4_Revolving_Credit', 'PAY_5_No_Transactions',
           'PAY_5_Pay_Duly', 'PAY_5_Revolving_Credit', 'PAY_6_No_Transactions',
           'PAY_6_Pay_Duly', 'PAY_6_Revolving_Credit'],
          dtype='object')




```python
#check for perfect collinearity
corr = df1.corr()
for i in range(len(corr)):
    corr.iloc[i,i] = 0
#corr[corr == 1] = 0
corr[corr.eq(1).any(1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>PAY_5</th>
      <th>PAY_6</th>
      <th>BILL_AMT1</th>
      <th>...</th>
      <th>PAY_3_Revolving_Credit</th>
      <th>PAY_4_No_Transactions</th>
      <th>PAY_4_Pay_Duly</th>
      <th>PAY_4_Revolving_Credit</th>
      <th>PAY_5_No_Transactions</th>
      <th>PAY_5_Pay_Duly</th>
      <th>PAY_5_Revolving_Credit</th>
      <th>PAY_6_No_Transactions</th>
      <th>PAY_6_Pay_Duly</th>
      <th>PAY_6_Revolving_Credit</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 47 columns</p>
</div>




```python
size = df1.shape
print("Data has {} Columns and {} Rows".format(size[1], size[0]))
```

    Data has 47 Columns and 26245 Rows
    

### Train Test Split

Before we conduct feature selection and model selection, we split the data using a train test split according to the project description.


```python
from sklearn.metrics import *
from sklearn.model_selection import *
```


```python
#using holdout sampling for train test split using seed 123
np.random.seed(123) 
ft = df1.drop("Y", axis = 1)
target = df1["Y"]
X_train,X_test,y_train,y_test = train_test_split(ft,target,test_size=1/3)
```

### Filter method for feature selection
The filter method for feature selection entails selecting relevant attributes before moving on to learning phase.
We will utitlise univariate feature selection to reduce the features to the fewer more significant attributes. 


```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

selector = SelectKBest( score_func = chi2, k=10)
selector.fit(X_train, y_train)
np.set_printoptions(precision=10)
chi2data = pd.DataFrame(selector.scores_)
chi2data["pval"] = 1 - stats.chi2.cdf(chi2data, 43)
chi2data.index = X_train.columns

print("Significant values are:")
print(chi2data[chi2data["pval"] < 0.05])

cols = chi2data[chi2data["pval"] < 0.05].index
X_train_filter = X_train[cols]
X_test_filter = X_test[cols]
```

    Significant values are:
                                      0          pval
    LIMIT_BAL                 82.306062  2.883753e-04
    PAY_0                   4279.993739  0.000000e+00
    PAY_2                   3557.072141  0.000000e+00
    PAY_3                   2766.119390  0.000000e+00
    PAY_4                   2736.965012  0.000000e+00
    PAY_5                   2587.002458  0.000000e+00
    PAY_6                   2240.874786  0.000000e+00
    PAY_0_No_Transactions     76.858872  1.147939e-03
    PAY_0_Revolving_Credit   480.805794  0.000000e+00
    PAY_2_Pay_Duly            75.283344  1.684018e-03
    PAY_2_Revolving_Credit   229.527990  0.000000e+00
    PAY_3_Pay_Duly            86.995856  8.229607e-05
    PAY_3_Revolving_Credit   121.059740  2.357071e-09
    PAY_4_Pay_Duly            79.449207  6.014800e-04
    PAY_4_Revolving_Credit    82.276504  2.906105e-04
    PAY_5_Pay_Duly            63.330298  2.338310e-02
    PAY_5_Revolving_Credit    64.659773  1.792035e-02
    

## Model Selection

In this part, we will fit machine learning models learnt in BT2101 to this classification problem, and pick the model that can produce the best results.

We will be attempting to fit the following models:


- Decision Tree 
- Logistic Regression
- Support Vector Machine
- Neural Network

To make things easier, we define a get_roc function that will plot an ROC curve for all the models we evaluate, and a confusion matrix function.



```python
def get_roc(model, y_test, X_test, name):
    try:
        fpr = roc_curve(y_test,model.predict_proba(X_test)[:,1])[0]
        tpr = roc_curve(y_test,model.predict_proba(X_test)[:,1])[1]
        thresholds = roc_curve(y_test,model.predict_proba(X_test)[:,1])[2]
    except:
        fpr = roc_curve(y_test,model.predict(X_test))[0]
        tpr = roc_curve(y_test,model.predict(X_test))[1]
        thresholds = roc_curve(y_test,model.predict(X_test))[2]
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for ' + name)
    plt.plot(fpr,tpr,label='ROC curve (AUC = %0.2f)' % (auc(fpr, tpr)))
    plt.legend(loc="lower right")
    
    #find- best threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Optimal Threshold: " + str(optimal_threshold))
    
    plt.show()
    
    return auc(fpr, tpr)
```


```python
def get_optimal(model, y_test, X_test, name):
    try:
        fpr = roc_curve(y_test,model.predict_proba(X_test)[:,1])[0]
        tpr = roc_curve(y_test,model.predict_proba(X_test)[:,1])[1]
        thresholds = roc_curve(y_test,model.predict_proba(X_test)[:,1])[2]
    except:
        fpr = roc_curve(y_test,model.predict(X_test))[0]
        tpr = roc_curve(y_test,model.predict(X_test))[1]
        thresholds = roc_curve(y_test,model.predict(X_test))[2]
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold            
```


```python
def confusion(y_test, predictions, name):
    conf = pd.crosstab(y_test,predictions, rownames=['Actual'], colnames=['Predicted'])
    print("Of " + str(conf[0][1] + conf[1][1]) + " Defaulters, the " + name + " identified " + str(conf[1][1])) 
    return conf
```

### Evaluation 
We will select the model based on the model evaluation. The key metrics we will compute are:

1. Accuracy
2. Recall
3. AUROC

Because of the nature of a default detection problem, we would like to prioritise **recall** for defaults. 
This means we will place more importance in correctly identifying a defaulter than avoiding misclassifying a non-defaulter. (Assumming that the bank loses more money when lending to a defaulter than not lending to a non-defaulter)

However, simply predicting every data point as a defaulter will give us 100% recall. We have to also consider accuracy and AUROC to get a better idea of how our model performs.



```python
evaluation = pd.DataFrame(columns=['Model', 'F1-1', 'AUROC'])
```

###  Decision Trees

#### Theory:
The decision tree algorithm aims to recursively split the data points in the training set until the data points are completely separated or well separated. At each iteration, the tree splits the datasets by the feature(s) that give the maximum reduction in heterogeneity, which is calculated by a heterogeneity index.

Below is a binary decision tree that has been split for a few iterations.

![image.png](https://elf11.github.io/images/decisionTree.png)

Since the target for this project is binary (fraud = yes or no) we will be building a binary decision tree, using the the GINI Index as the Heterogeneity index. The GINI is given by:

![image.png](https://miro.medium.com/max/664/1*otdoiyIwxJI-UV0ukkyutw.png)

The GINI index measures how heterogenous a single node is (0 being completely homogenous and 1 being heterogenous). For each possible split, we will calculate the *weighted sum* of the GINI indices of the child nodes, and choose the split that results in the maximum information gain. i.e. reduction in the weighted sum of the GINI Index.

#### Training
We will now construct a simple decision tree using the GINI index.


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=None, splitter='best')




```python
get_roc(tree, y_train, X_train, "Decision Tree (GINI)")
print(classification_report(y_train, tree.predict(X_train)))
```

    Optimal Threshold: 0.3333333333333333
    


![png](defaults_files/defaults_61_1.png)


                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00     13442
               1       1.00      1.00      1.00      4054
    
        accuracy                           1.00     17496
       macro avg       1.00      1.00      1.00     17496
    weighted avg       1.00      1.00      1.00     17496
    
    

The training set accuracy is 1, which means the datapoints are completely separated by the decision tree. We evaluate on the test set below.


```python
confusion(y_test, tree.predict(X_test), "Decision Tree (GINI)")
```

    Of 1987 Defaulters, the Decision Tree (GINI) identified 809
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Actual</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5482</td>
      <td>1280</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1178</td>
      <td>809</td>
    </tr>
  </tbody>
</table>
</div>




```python
auroc = get_roc(tree, y_test, X_test, "Decision Tree (GINI)")
print(classification_report(y_test, tree.predict(X_test)))
```

    Optimal Threshold: 0.5
    


![png](defaults_files/defaults_64_1.png)


                  precision    recall  f1-score   support
    
               0       0.82      0.81      0.82      6762
               1       0.39      0.41      0.40      1987
    
        accuracy                           0.72      8749
       macro avg       0.61      0.61      0.61      8749
    weighted avg       0.72      0.72      0.72      8749
    
    


```python
tree2 = DecisionTreeClassifier(criterion = "entropy")
tree2.fit(X_train, y_train)
confusion(y_test, tree2.predict(X_test), "Decision Tree (Entropy)")
```

    Of 1987 Defaulters, the Decision Tree (Entropy) identified 831
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Actual</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5509</td>
      <td>1253</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1156</td>
      <td>831</td>
    </tr>
  </tbody>
</table>
</div>




```python
get_roc(tree2, y_test, X_test, "Decision Tree (Entropy)")
print(classification_report(y_test, tree2.predict(X_test)))
```

    Optimal Threshold: 0.5
    


![png](defaults_files/defaults_66_1.png)


                  precision    recall  f1-score   support
    
               0       0.83      0.81      0.82      6762
               1       0.40      0.42      0.41      1987
    
        accuracy                           0.72      8749
       macro avg       0.61      0.62      0.61      8749
    weighted avg       0.73      0.72      0.73      8749
    
    

There is negligible difference in using GINI or Entropy for decision trees. For the sake of simplicity, we will use GINI for the ensemble methods.

### Random Forest Classifier

#### Theory
Random Forest is an ensemble method for the decision tree algorithm. It works by randomly choosing different features and data points to train multiple trees (that is, to form a forest) - and the resulting prediction is decided by the votes from all the trees. 

Decision Trees are prone to overfitting on the training data, which reduces the performance on the test set. Random Forest mitigates this by training multiple trees. Random Forest is a form of bagging ensemble where the trees are trained concurrently. 

#### Training
To keep things consistent, our Random Forest classifier will also use the GINI Coefficient.




```python
from sklearn.ensemble import RandomForestClassifier
randf = RandomForestClassifier(n_estimators=200)
```


```python
randf.fit(X_train, y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=200,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)




```python
print(classification_report(y_train, randf.predict(X_train)))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00     13442
               1       1.00      1.00      1.00      4054
    
        accuracy                           1.00     17496
       macro avg       1.00      1.00      1.00     17496
    weighted avg       1.00      1.00      1.00     17496
    
    

The training set has also been 100% correctly classified by the random forest model. Evaluating with the test set:


```python
confusion(y_test, randf.predict(X_test), "Decision Tree (Random Forest)")
```

    Of 1987 Defaulters, the Decision Tree (Random Forest) identified 713
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Actual</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>6371</td>
      <td>391</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1274</td>
      <td>713</td>
    </tr>
  </tbody>
</table>
</div>




```python
auroc_rf = get_roc(randf, y_test, X_test, "Decision Tree (Random Forest)")
print(classification_report(y_test, randf.predict(X_test)))
```

    Optimal Threshold: 0.27
    


![png](defaults_files/defaults_74_1.png)


                  precision    recall  f1-score   support
    
               0       0.83      0.94      0.88      6762
               1       0.65      0.36      0.46      1987
    
        accuracy                           0.81      8749
       macro avg       0.74      0.65      0.67      8749
    weighted avg       0.79      0.81      0.79      8749
    
    

The random forest ensemble performs much better than the decision tree alone. The accuracy and AUROC are both superior to the decision tree alone.

### Gradient Boosted Trees Classifier

#### Theory
In this part we train a gradient boosted trees classifier. It is a boosting ensemble method for decision trees, which means that the trees are trained consecutively, where each new tree added is trained to correct the error from the previous tree.
 
#### Training
For consistency our xgBoost ensemble will use n_estimators = 300 as we have done for the random forest ensemble.


```python
from sklearn.ensemble import GradientBoostingClassifier
xgb = GradientBoostingClassifier(n_estimators=300, max_depth = 4)
xgb.fit(X_train, y_train)
```




    GradientBoostingClassifier(criterion='friedman_mse', init=None,
                               learning_rate=0.1, loss='deviance', max_depth=4,
                               max_features=None, max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=300,
                               n_iter_no_change=None, presort='auto',
                               random_state=None, subsample=1.0, tol=0.0001,
                               validation_fraction=0.1, verbose=0,
                               warm_start=False)




```python
print(classification_report(y_train, xgb.predict(X_train)))
```

                  precision    recall  f1-score   support
    
               0       0.86      0.96      0.91     13442
               1       0.79      0.46      0.58      4054
    
        accuracy                           0.85     17496
       macro avg       0.82      0.71      0.74     17496
    weighted avg       0.84      0.85      0.83     17496
    
    

We observe that the ensemble did not fully separate the data in the training set. (The default maximum depth is 3, so that might be a factor). Evaluating on the test set,


```python
confusion(y_test, xgb.predict(X_test), "Decision Tree (Gradient Boosted Trees)")
```

    Of 1987 Defaulters, the Decision Tree (Gradient Boosted Trees) identified 717
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Actual</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>6381</td>
      <td>381</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1270</td>
      <td>717</td>
    </tr>
  </tbody>
</table>
</div>




```python
auroc = get_roc(xgb, y_test, X_test, "Decision Tree (XGBoost)")
print(classification_report(y_test, xgb.predict(X_test)))
```

    Optimal Threshold: 0.24738247273049666
    


![png](defaults_files/defaults_81_1.png)


From both the accuracy metrics and the AUROC, we observe that the gradient boosted tree performs similarly to the random forest classifier. We will choose Random Forest as our model of choice using the decision tree algorithm.


```python
evaluation.loc[0] = (["Decision Trees - Random Forest" , 
                      classification_report(y_test, randf.predict(X_test), output_dict = True)["1"]["f1-score"],
                      auroc_rf])
```


```python
evaluation
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>F1-1</th>
      <th>AUROC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Decision Trees - Random Forest</td>
      <td>0.461339</td>
      <td>0.768458</td>
    </tr>
  </tbody>
</table>
</div>



### Logistic Regression

#### Theory
Logistic regression is a regression technnique used to predict binary target variables. It works on the same principles as a linear regression model. 

Our binary target (default vs non-default) can be expressed in terms of odds of defaulting, which is the ratio of the probability of default and probability of non-default. 

In the logistic regression model, we log the odds (log-odds) and equate it to a weighted sum of regressors.

![image.png](https://wikimedia.org/api/rest_v1/media/math/render/svg/4a5e86f014eb1f0744e280eb0d68485cb8c0a6c3)

We then find weights for the regressors that best fits the data. Since the binary target (default or not) follows a bernoulli distribution, each data point has the following probability distribution function:

![image.png](https://wikimedia.org/api/rest_v1/media/math/render/svg/614e0c64d59f0ff2e926deafcb2de6e502394fac)

We would like to update p for each data point such that the log product (joint probability) of the above function for all data points is maximised. In other words, we are maximising the log-likelihood function.

The logistic regression equation produces a "squashed" curve like the one below. We then pick a cutoff value for the y axis to classify a data point as 0 (non-default) or 1 (default).

![image.png](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1280px-Logistic-curve.svg.png)


#### Training
We will adopt a top-down approach for training our logistic regression model, i.e. include all regressors first and then remove the most insignificant ones at each iteration to achieve the best fit.


```python
import statsmodels.api as sm
```


```python
glm = sm.Logit(y_train,X_train).fit()
glm.summary()
```

    Warning: Maximum number of iterations has been exceeded.
             Current function value: 0.444770
             Iterations: 35
    

    C:\Users\reonh\Anaconda3\lib\site-packages\statsmodels\base\model.py:512: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      "Check mle_retvals", ConvergenceWarning)
    




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>Y</td>        <th>  No. Observations:  </th>  <td> 17496</td> 
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td> 17450</td> 
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>    45</td> 
</tr>
<tr>
  <th>Date:</th>            <td>Fri, 22 Nov 2019</td> <th>  Pseudo R-squ.:     </th>  <td>0.1784</td> 
</tr>
<tr>
  <th>Time:</th>                <td>00:13:23</td>     <th>  Log-Likelihood:    </th> <td> -7781.7</td>
</tr>
<tr>
  <th>converged:</th>             <td>False</td>      <th>  LL-Null:           </th> <td> -9471.2</td>
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td> 0.000</td> 
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>               <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>LIMIT_BAL</th>              <td>   -0.8737</td> <td>    0.115</td> <td>   -7.605</td> <td> 0.000</td> <td>   -1.099</td> <td>   -0.649</td>
</tr>
<tr>
  <th>SEX</th>                    <td>   -0.0964</td> <td>    0.041</td> <td>   -2.343</td> <td> 0.019</td> <td>   -0.177</td> <td>   -0.016</td>
</tr>
<tr>
  <th>AGE</th>                    <td>    0.2097</td> <td>    0.100</td> <td>    2.095</td> <td> 0.036</td> <td>    0.013</td> <td>    0.406</td>
</tr>
<tr>
  <th>PAY_0</th>                  <td>    0.6116</td> <td>    0.058</td> <td>   10.521</td> <td> 0.000</td> <td>    0.498</td> <td>    0.726</td>
</tr>
<tr>
  <th>PAY_2</th>                  <td>   -0.5528</td> <td>    0.096</td> <td>   -5.763</td> <td> 0.000</td> <td>   -0.741</td> <td>   -0.365</td>
</tr>
<tr>
  <th>PAY_3</th>                  <td>   -0.2063</td> <td>    0.124</td> <td>   -1.662</td> <td> 0.096</td> <td>   -0.450</td> <td>    0.037</td>
</tr>
<tr>
  <th>PAY_4</th>                  <td>   -0.2327</td> <td>    0.160</td> <td>   -1.452</td> <td> 0.146</td> <td>   -0.547</td> <td>    0.081</td>
</tr>
<tr>
  <th>PAY_5</th>                  <td>   -0.0302</td> <td>    0.181</td> <td>   -0.166</td> <td> 0.868</td> <td>   -0.385</td> <td>    0.325</td>
</tr>
<tr>
  <th>PAY_6</th>                  <td>    0.4319</td> <td>    0.153</td> <td>    2.825</td> <td> 0.005</td> <td>    0.132</td> <td>    0.731</td>
</tr>
<tr>
  <th>BILL_AMT1</th>              <td>   -1.9057</td> <td>    0.554</td> <td>   -3.442</td> <td> 0.001</td> <td>   -2.991</td> <td>   -0.821</td>
</tr>
<tr>
  <th>BILL_AMT2</th>              <td>    1.1700</td> <td>    0.784</td> <td>    1.493</td> <td> 0.135</td> <td>   -0.366</td> <td>    2.706</td>
</tr>
<tr>
  <th>BILL_AMT3</th>              <td>    1.9680</td> <td>    0.729</td> <td>    2.700</td> <td> 0.007</td> <td>    0.540</td> <td>    3.396</td>
</tr>
<tr>
  <th>BILL_AMT4</th>              <td>   -0.4328</td> <td>    0.727</td> <td>   -0.595</td> <td> 0.552</td> <td>   -1.858</td> <td>    0.992</td>
</tr>
<tr>
  <th>BILL_AMT5</th>              <td>   -0.3910</td> <td>    0.882</td> <td>   -0.443</td> <td> 0.658</td> <td>   -2.120</td> <td>    1.338</td>
</tr>
<tr>
  <th>BILL_AMT6</th>              <td>    0.2306</td> <td>    0.800</td> <td>    0.288</td> <td> 0.773</td> <td>   -1.338</td> <td>    1.799</td>
</tr>
<tr>
  <th>PAY_AMT1</th>               <td>   -1.2427</td> <td>    0.308</td> <td>   -4.041</td> <td> 0.000</td> <td>   -1.845</td> <td>   -0.640</td>
</tr>
<tr>
  <th>PAY_AMT2</th>               <td>   -1.8767</td> <td>    0.389</td> <td>   -4.823</td> <td> 0.000</td> <td>   -2.639</td> <td>   -1.114</td>
</tr>
<tr>
  <th>PAY_AMT3</th>               <td>   -0.4002</td> <td>    0.299</td> <td>   -1.339</td> <td> 0.181</td> <td>   -0.986</td> <td>    0.186</td>
</tr>
<tr>
  <th>PAY_AMT4</th>               <td>   -0.5031</td> <td>    0.293</td> <td>   -1.715</td> <td> 0.086</td> <td>   -1.078</td> <td>    0.072</td>
</tr>
<tr>
  <th>PAY_AMT5</th>               <td>   -0.7629</td> <td>    0.295</td> <td>   -2.589</td> <td> 0.010</td> <td>   -1.341</td> <td>   -0.185</td>
</tr>
<tr>
  <th>PAY_AMT6</th>               <td>   -0.6658</td> <td>    0.266</td> <td>   -2.504</td> <td> 0.012</td> <td>   -1.187</td> <td>   -0.145</td>
</tr>
<tr>
  <th>MISSING-EDU</th>            <td>  -14.2753</td> <td> 1898.465</td> <td>   -0.008</td> <td> 0.994</td> <td>-3735.198</td> <td> 3706.648</td>
</tr>
<tr>
  <th>GRAD</th>                   <td>    1.3518</td> <td>    0.220</td> <td>    6.148</td> <td> 0.000</td> <td>    0.921</td> <td>    1.783</td>
</tr>
<tr>
  <th>UNI</th>                    <td>    1.3056</td> <td>    0.219</td> <td>    5.971</td> <td> 0.000</td> <td>    0.877</td> <td>    1.734</td>
</tr>
<tr>
  <th>HS</th>                     <td>    1.2342</td> <td>    0.223</td> <td>    5.547</td> <td> 0.000</td> <td>    0.798</td> <td>    1.670</td>
</tr>
<tr>
  <th>MISSING-MS</th>             <td>  -30.7439</td> <td> 1.14e+06</td> <td> -2.7e-05</td> <td> 1.000</td> <td>-2.23e+06</td> <td> 2.23e+06</td>
</tr>
<tr>
  <th>MARRIED</th>                <td>    0.0794</td> <td>    0.177</td> <td>    0.449</td> <td> 0.653</td> <td>   -0.267</td> <td>    0.426</td>
</tr>
<tr>
  <th>SINGLE</th>                 <td>   -0.1024</td> <td>    0.177</td> <td>   -0.577</td> <td> 0.564</td> <td>   -0.450</td> <td>    0.245</td>
</tr>
<tr>
  <th>PAY_0_No_Transactions</th>  <td>   -0.1746</td> <td>    0.123</td> <td>   -1.415</td> <td> 0.157</td> <td>   -0.416</td> <td>    0.067</td>
</tr>
<tr>
  <th>PAY_0_Pay_Duly</th>         <td>    0.0483</td> <td>    0.120</td> <td>    0.402</td> <td> 0.688</td> <td>   -0.187</td> <td>    0.284</td>
</tr>
<tr>
  <th>PAY_0_Revolving_Credit</th> <td>   -0.9702</td> <td>    0.135</td> <td>   -7.181</td> <td> 0.000</td> <td>   -1.235</td> <td>   -0.705</td>
</tr>
<tr>
  <th>PAY_2_No_Transactions</th>  <td>   -1.4826</td> <td>    0.233</td> <td>   -6.359</td> <td> 0.000</td> <td>   -1.940</td> <td>   -1.026</td>
</tr>
<tr>
  <th>PAY_2_Pay_Duly</th>         <td>   -1.3804</td> <td>    0.221</td> <td>   -6.244</td> <td> 0.000</td> <td>   -1.814</td> <td>   -0.947</td>
</tr>
<tr>
  <th>PAY_2_Revolving_Credit</th> <td>   -0.7926</td> <td>    0.226</td> <td>   -3.514</td> <td> 0.000</td> <td>   -1.235</td> <td>   -0.350</td>
</tr>
<tr>
  <th>PAY_3_No_Transactions</th>  <td>   -0.6881</td> <td>    0.297</td> <td>   -2.317</td> <td> 0.021</td> <td>   -1.270</td> <td>   -0.106</td>
</tr>
<tr>
  <th>PAY_3_Pay_Duly</th>         <td>   -0.7811</td> <td>    0.272</td> <td>   -2.869</td> <td> 0.004</td> <td>   -1.315</td> <td>   -0.247</td>
</tr>
<tr>
  <th>PAY_3_Revolving_Credit</th> <td>   -0.7137</td> <td>    0.261</td> <td>   -2.740</td> <td> 0.006</td> <td>   -1.224</td> <td>   -0.203</td>
</tr>
<tr>
  <th>PAY_4_No_Transactions</th>  <td>   -0.9092</td> <td>    0.360</td> <td>   -2.529</td> <td> 0.011</td> <td>   -1.614</td> <td>   -0.205</td>
</tr>
<tr>
  <th>PAY_4_Pay_Duly</th>         <td>   -0.9199</td> <td>    0.341</td> <td>   -2.699</td> <td> 0.007</td> <td>   -1.588</td> <td>   -0.252</td>
</tr>
<tr>
  <th>PAY_4_Revolving_Credit</th> <td>   -0.8088</td> <td>    0.331</td> <td>   -2.442</td> <td> 0.015</td> <td>   -1.458</td> <td>   -0.160</td>
</tr>
<tr>
  <th>PAY_5_No_Transactions</th>  <td>   -0.0741</td> <td>    0.401</td> <td>   -0.185</td> <td> 0.853</td> <td>   -0.860</td> <td>    0.711</td>
</tr>
<tr>
  <th>PAY_5_Pay_Duly</th>         <td>   -0.2557</td> <td>    0.386</td> <td>   -0.663</td> <td> 0.507</td> <td>   -1.011</td> <td>    0.500</td>
</tr>
<tr>
  <th>PAY_5_Revolving_Credit</th> <td>   -0.2701</td> <td>    0.376</td> <td>   -0.718</td> <td> 0.473</td> <td>   -1.008</td> <td>    0.467</td>
</tr>
<tr>
  <th>PAY_6_No_Transactions</th>  <td>    0.6784</td> <td>    0.335</td> <td>    2.025</td> <td> 0.043</td> <td>    0.022</td> <td>    1.335</td>
</tr>
<tr>
  <th>PAY_6_Pay_Duly</th>         <td>    0.7000</td> <td>    0.328</td> <td>    2.134</td> <td> 0.033</td> <td>    0.057</td> <td>    1.343</td>
</tr>
<tr>
  <th>PAY_6_Revolving_Credit</th> <td>    0.5159</td> <td>    0.320</td> <td>    1.615</td> <td> 0.106</td> <td>   -0.110</td> <td>    1.142</td>
</tr>
</table>




```python
print(classification_report(y_train,list(glm.predict(X_train)>0.5)))
```

                  precision    recall  f1-score   support
    
               0       0.83      0.95      0.88     13442
               1       0.67      0.36      0.47      4054
    
        accuracy                           0.81     17496
       macro avg       0.75      0.65      0.68     17496
    weighted avg       0.79      0.81      0.79     17496
    
    

The logisitc model with all features performs average on both the train and test set with an accuracy of about 0.8 but recall and f1 are still below 0.5. We will now try removing all the insignificant features to see how that affects the model performance.


```python
#remove the most insignificant attribute, and retrain
train_log =  X_train.copy()
glm = sm.Logit(y_train,train_log).fit()
while max(glm.pvalues) > 0.01:
    least =  glm.pvalues[glm.pvalues == max(glm.pvalues)].index[0]
    train_log = train_log.drop(least,axis = 1)
    glm = sm.Logit(y_train,train_log).fit()
glm.summary()   
```

    Warning: Maximum number of iterations has been exceeded.
             Current function value: 0.444770
             Iterations: 35
    

    C:\Users\reonh\Anaconda3\lib\site-packages\statsmodels\base\model.py:512: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      "Check mle_retvals", ConvergenceWarning)
    

    Warning: Maximum number of iterations has been exceeded.
             Current function value: 0.445360
             Iterations: 35
    Optimization terminated successfully.
             Current function value: 0.445386
             Iterations 7
    Optimization terminated successfully.
             Current function value: 0.445386
             Iterations 7
    Optimization terminated successfully.
             Current function value: 0.445387
             Iterations 7
    

    C:\Users\reonh\Anaconda3\lib\site-packages\statsmodels\base\model.py:512: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      "Check mle_retvals", ConvergenceWarning)
    

    Optimization terminated successfully.
             Current function value: 0.445388
             Iterations 7
    Optimization terminated successfully.
             Current function value: 0.445392
             Iterations 7
    Optimization terminated successfully.
             Current function value: 0.445397
             Iterations 7
    Optimization terminated successfully.
             Current function value: 0.445410
             Iterations 7
    Optimization terminated successfully.
             Current function value: 0.445455
             Iterations 7
    Optimization terminated successfully.
             Current function value: 0.445512
             Iterations 7
    Optimization terminated successfully.
             Current function value: 0.445596
             Iterations 7
    Optimization terminated successfully.
             Current function value: 0.445680
             Iterations 7
    Optimization terminated successfully.
             Current function value: 0.445770
             Iterations 7
    Optimization terminated successfully.
             Current function value: 0.445853
             Iterations 7
    Optimization terminated successfully.
             Current function value: 0.445877
             Iterations 7
    Optimization terminated successfully.
             Current function value: 0.445963
             Iterations 7
    Optimization terminated successfully.
             Current function value: 0.446090
             Iterations 7
    Optimization terminated successfully.
             Current function value: 0.446288
             Iterations 7
    




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>Y</td>        <th>  No. Observations:  </th>  <td> 17496</td> 
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td> 17468</td> 
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>    27</td> 
</tr>
<tr>
  <th>Date:</th>            <td>Fri, 22 Nov 2019</td> <th>  Pseudo R-squ.:     </th>  <td>0.1756</td> 
</tr>
<tr>
  <th>Time:</th>                <td>00:14:16</td>     <th>  Log-Likelihood:    </th> <td> -7808.3</td>
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -9471.2</td>
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th>  <td> 0.000</td> 
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>               <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>LIMIT_BAL</th>              <td>   -0.8984</td> <td>    0.113</td> <td>   -7.922</td> <td> 0.000</td> <td>   -1.121</td> <td>   -0.676</td>
</tr>
<tr>
  <th>SEX</th>                    <td>   -0.1153</td> <td>    0.041</td> <td>   -2.847</td> <td> 0.004</td> <td>   -0.195</td> <td>   -0.036</td>
</tr>
<tr>
  <th>PAY_0</th>                  <td>    0.6189</td> <td>    0.037</td> <td>   16.520</td> <td> 0.000</td> <td>    0.545</td> <td>    0.692</td>
</tr>
<tr>
  <th>PAY_2</th>                  <td>   -0.5692</td> <td>    0.088</td> <td>   -6.463</td> <td> 0.000</td> <td>   -0.742</td> <td>   -0.397</td>
</tr>
<tr>
  <th>PAY_3</th>                  <td>   -0.2710</td> <td>    0.082</td> <td>   -3.313</td> <td> 0.001</td> <td>   -0.431</td> <td>   -0.111</td>
</tr>
<tr>
  <th>PAY_6</th>                  <td>    0.2151</td> <td>    0.031</td> <td>    6.899</td> <td> 0.000</td> <td>    0.154</td> <td>    0.276</td>
</tr>
<tr>
  <th>BILL_AMT1</th>              <td>   -1.3934</td> <td>    0.368</td> <td>   -3.784</td> <td> 0.000</td> <td>   -2.115</td> <td>   -0.672</td>
</tr>
<tr>
  <th>BILL_AMT3</th>              <td>    2.0154</td> <td>    0.435</td> <td>    4.638</td> <td> 0.000</td> <td>    1.164</td> <td>    2.867</td>
</tr>
<tr>
  <th>PAY_AMT1</th>               <td>   -1.2565</td> <td>    0.287</td> <td>   -4.371</td> <td> 0.000</td> <td>   -1.820</td> <td>   -0.693</td>
</tr>
<tr>
  <th>PAY_AMT2</th>               <td>   -2.1865</td> <td>    0.376</td> <td>   -5.816</td> <td> 0.000</td> <td>   -2.923</td> <td>   -1.450</td>
</tr>
<tr>
  <th>PAY_AMT5</th>               <td>   -0.8702</td> <td>    0.265</td> <td>   -3.279</td> <td> 0.001</td> <td>   -1.390</td> <td>   -0.350</td>
</tr>
<tr>
  <th>PAY_AMT6</th>               <td>   -0.7982</td> <td>    0.266</td> <td>   -3.000</td> <td> 0.003</td> <td>   -1.320</td> <td>   -0.277</td>
</tr>
<tr>
  <th>GRAD</th>                   <td>    1.3465</td> <td>    0.175</td> <td>    7.687</td> <td> 0.000</td> <td>    1.003</td> <td>    1.690</td>
</tr>
<tr>
  <th>UNI</th>                    <td>    1.2982</td> <td>    0.174</td> <td>    7.462</td> <td> 0.000</td> <td>    0.957</td> <td>    1.639</td>
</tr>
<tr>
  <th>HS</th>                     <td>    1.2384</td> <td>    0.178</td> <td>    6.960</td> <td> 0.000</td> <td>    0.890</td> <td>    1.587</td>
</tr>
<tr>
  <th>MARRIED</th>                <td>    0.2359</td> <td>    0.042</td> <td>    5.643</td> <td> 0.000</td> <td>    0.154</td> <td>    0.318</td>
</tr>
<tr>
  <th>PAY_0_Revolving_Credit</th> <td>   -0.9811</td> <td>    0.093</td> <td>  -10.583</td> <td> 0.000</td> <td>   -1.163</td> <td>   -0.799</td>
</tr>
<tr>
  <th>PAY_2_No_Transactions</th>  <td>   -1.5901</td> <td>    0.220</td> <td>   -7.230</td> <td> 0.000</td> <td>   -2.021</td> <td>   -1.159</td>
</tr>
<tr>
  <th>PAY_2_Pay_Duly</th>         <td>   -1.4026</td> <td>    0.200</td> <td>   -7.010</td> <td> 0.000</td> <td>   -1.795</td> <td>   -1.010</td>
</tr>
<tr>
  <th>PAY_2_Revolving_Credit</th> <td>   -0.8163</td> <td>    0.202</td> <td>   -4.051</td> <td> 0.000</td> <td>   -1.211</td> <td>   -0.421</td>
</tr>
<tr>
  <th>PAY_3_No_Transactions</th>  <td>   -0.8432</td> <td>    0.228</td> <td>   -3.701</td> <td> 0.000</td> <td>   -1.290</td> <td>   -0.397</td>
</tr>
<tr>
  <th>PAY_3_Pay_Duly</th>         <td>   -0.8926</td> <td>    0.196</td> <td>   -4.566</td> <td> 0.000</td> <td>   -1.276</td> <td>   -0.509</td>
</tr>
<tr>
  <th>PAY_3_Revolving_Credit</th> <td>   -0.8227</td> <td>    0.179</td> <td>   -4.586</td> <td> 0.000</td> <td>   -1.174</td> <td>   -0.471</td>
</tr>
<tr>
  <th>PAY_4_No_Transactions</th>  <td>   -0.4537</td> <td>    0.143</td> <td>   -3.172</td> <td> 0.002</td> <td>   -0.734</td> <td>   -0.173</td>
</tr>
<tr>
  <th>PAY_4_Pay_Duly</th>         <td>   -0.5711</td> <td>    0.107</td> <td>   -5.328</td> <td> 0.000</td> <td>   -0.781</td> <td>   -0.361</td>
</tr>
<tr>
  <th>PAY_4_Revolving_Credit</th> <td>   -0.4353</td> <td>    0.075</td> <td>   -5.806</td> <td> 0.000</td> <td>   -0.582</td> <td>   -0.288</td>
</tr>
<tr>
  <th>PAY_6_No_Transactions</th>  <td>    0.3028</td> <td>    0.089</td> <td>    3.399</td> <td> 0.001</td> <td>    0.128</td> <td>    0.477</td>
</tr>
<tr>
  <th>PAY_6_Pay_Duly</th>         <td>    0.2489</td> <td>    0.078</td> <td>    3.197</td> <td> 0.001</td> <td>    0.096</td> <td>    0.402</td>
</tr>
</table>




```python
count = len(glm.pvalues.index)
print(str(count) + " Columns left:")
print(glm.pvalues.index)
```

    28 Columns left:
    Index(['LIMIT_BAL', 'SEX', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_6', 'BILL_AMT1',
           'BILL_AMT3', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT5', 'PAY_AMT6', 'GRAD',
           'UNI', 'HS', 'MARRIED', 'PAY_0_Revolving_Credit',
           'PAY_2_No_Transactions', 'PAY_2_Pay_Duly', 'PAY_2_Revolving_Credit',
           'PAY_3_No_Transactions', 'PAY_3_Pay_Duly', 'PAY_3_Revolving_Credit',
           'PAY_4_No_Transactions', 'PAY_4_Pay_Duly', 'PAY_4_Revolving_Credit',
           'PAY_6_No_Transactions', 'PAY_6_Pay_Duly'],
          dtype='object')
    


```python
print(classification_report(y_test,list(glm.predict(X_test[glm.pvalues.index])>0.5)))
```

                  precision    recall  f1-score   support
    
               0       0.83      0.95      0.89      6762
               1       0.68      0.36      0.47      1987
    
        accuracy                           0.82      8749
       macro avg       0.76      0.65      0.68      8749
    weighted avg       0.80      0.82      0.79      8749
    
    

Since there is not much change to the model performance on both the train and test set when we reduce the features, we will use the reduced logistic regression model from this point onwards (Principle of Parsimony). 

We now Calculate the AUROC for the train set.


```python
optimal_log = get_optimal(glm, y_train, X_train[glm.pvalues.index], "Logistic Regression")
get_roc(glm, y_train, X_train[glm.pvalues.index], "Logistic Regression")
print(classification_report(y_test,list(glm.predict(X_test[glm.pvalues.index])> optimal_log)))
```

    Optimal Threshold: 0.21650864211883647
    


![png](defaults_files/defaults_94_1.png)


                  precision    recall  f1-score   support
    
               0       0.88      0.78      0.83      6762
               1       0.46      0.62      0.53      1987
    
        accuracy                           0.75      8749
       macro avg       0.67      0.70      0.68      8749
    weighted avg       0.78      0.75      0.76      8749
    
    

Since the optimal cut off was found to be ~0.22 using the training data, we used that as our cut off for our evaluation of the test set.

Unfortunately, the training accuracy has gone down when we used the optimal cutoff. However, accuracy may be misleading in a dataset like ours where most of the targets are non-defaults. 

The recall here is more important - detecting defaulters is more useful than detecting non-defaulters. With a higher recall, our model with lower cutoff is able to correctly catch more defaulters. We note that this increase in recall is greater than the increase in F-1.


Calculate the confusion matrices for both cut offs to better compare their performance.


```python
confusion(y_test,glm.predict(X_test[glm.pvalues.index])>optimal_log, "Logistic Regression")
```

    Of 1987 Defaulters, the Logistic Regression identified 1235
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>False</th>
      <th>True</th>
    </tr>
    <tr>
      <th>Actual</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5303</td>
      <td>1459</td>
    </tr>
    <tr>
      <td>1</td>
      <td>752</td>
      <td>1235</td>
    </tr>
  </tbody>
</table>
</div>




```python
confusion(y_test,glm.predict(X_test[glm.pvalues.index])>0.50, "Logistic Regression")
```

    Of 1987 Defaulters, the Logistic Regression identified 715
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>False</th>
      <th>True</th>
    </tr>
    <tr>
      <th>Actual</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>6421</td>
      <td>341</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1272</td>
      <td>715</td>
    </tr>
  </tbody>
</table>
</div>



It is evident that the lower cutoff is better able to detect defaults. 


```python
auroc = get_roc(glm, y_test, X_test[glm.pvalues.index], "Logistic Regression")
```

    Optimal Threshold: 0.24907536768337235
    


![png](defaults_files/defaults_100_1.png)



```python
evaluation.loc[1] = ["Logistic Regression (Optimal Cutoff)" , 
                     classification_report(y_test,list(glm.predict(X_test[glm.pvalues.index])>optimal_log), output_dict = True)["1"]["f1-score"],
                     auroc]
```


```python
evaluation
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>F1-1</th>
      <th>AUROC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Decision Trees - Random Forest</td>
      <td>0.461339</td>
      <td>0.768458</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Logistic Regression (Optimal Cutoff)</td>
      <td>0.527665</td>
      <td>0.765244</td>
    </tr>
  </tbody>
</table>
</div>



### Support Vector Machine
#### Theory
Support vector machines attempt to find an optimal hyperplane that is able to separate the two classes in n-dimensional space. This is done by finding the hyperplane that maximises the distance between itself and support vectors (data points that lie closest to the decision boundary).

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/617px-SVM_margin.png" width="340" height="340" align="center"/> 

Given a training dataset of form (X,Y), where X is a vector of attributes of the sample, and where Y are either 1 or -1, each indicating the class to which the point X belongs. We want to find the "maximum-margin hyperplane" that divides the group of points X which Y = 1 from the group of points for which Y = -1, which is defined so that the distance between the hyperplane and the nearest point X from either group is maximized.

Any hyperplane can be written as the set of points X satisfying

<table><tr>
<td>
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/90e0fa283c9e642c9c11b22da45efa30b06944a9" width="140" height="140" align="left"/> 
</td>
</tr></table>

where W is the (not necessarily normalized) normal vector to the hyperplane. This is much like Hesse normal form, except that W is not necessarily a unit vector. The parameter b/||W|| determines the offset of the hyperplane from the origin along the normal vector W.

#### Margin
A margin is a separation of line to the closest class points.
Very importrant characteristic of SVM classifier. SVM to core tries to achieve a good margin.
A good margin is one where this separation is larger for both the classes. Images below gives to visual example of good and bad margin. A good margin allows the points to be in their respective classes without crossing to other class.    

<table><tr>
<td> <img src="https://miro.medium.com/max/600/1*Ftns0ebfUHJDdpWt3Wvp-Q.png" width="940" height="940" align="left"//> </td>
<td> <img src="https://miro.medium.com/max/600/1*NbGV1iEtNuklACNUv74w7A.png" width="940" height="940" align="right"/> </td>
</tr></table>

Our goal is to maximize the margin. Among all possible hyperplanes meeting the constraints,  we will choose the hyperplane with the smallest ‖w‖ because it is the one which will have the biggest margin.

##### Hard Margin
If the training data is linearly separable, we can select two parallel hyperplanes that separate the two classes of data, so that the distance between them is as large as possible. The region bounded by these two hyperplanes is called the "margin", and the maximum-margin hyperplane is the hyperplane that lies halfway between them. With a normalized or standardized dataset, these hyperplanes can be described by an equation and we can put this together to get the optimization problem:

 Minimize ||W|| subject to:
<table><tr>   
<td><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/94c99827acb10edd809df63bb86ca1366f01a8ac" width="=240" height="240" align="right"/>
</td>
</tr></table>

##### Soft Margin
Often, the data are not linearly separable. Thus, to extend SVM to cases in which the data are not linearly separable, we introduce the hinge loss function,
<table><tr>   
<td><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/f5f7d77f3d46cac51fbac58545aa1a1a183fdf7f" width="=240" height="240" align="right"/>
</td>
</tr></table>

This function is zero if the constraint in (1) is satisfied, in other words, if Xlies on the correct side of the margin. For data on the wrong side of the margin, the function's value is proportional to the distance from the margin.

We then wish to minimize
<table><tr>   
<td><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/579f80b069f186f5a0013b11f90f32833ff8c681" width="=240" height="240" align="right"/>
</td>
</tr></table>

where the parameter lambda determines the trade-off between increasing the margin size and ensuring that the X lie on the correct side of the margin. Thus, for sufficiently small values of lambda , the second term in the loss function will become negligible, hence, it will behave similar to the hard-margin SVM, if the input data are linearly classifiable, but will still learn if a classification rule is viable or not.

#### Computing SVM classifier
We focus on the soft-margin classifier since, as noted above, choosing a sufficiently small value for lambda  yields the hard-margin classifier for linearly classifiable input data. The classical approach, which involves reducing (2) to a quadratic programming problem, is detailed below.

##### Primal
Minimizing (2) can be rewritten as a constrained optimization problem with a differentiable objective function in the following way.

We can rewrite the optimization problem as follows
<table><tr>   
<td><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/71501a21527b0f375eb349fdaf95f33a78b1db6d" width="=240" height="240" align="right"/>
</td>
</tr></table>

This is called the primal problem.

##### Dual
By solving for the Lagrangian dual of the above problem, one obtains the simplified problem
<table><tr>   
<td><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/9c89851fa2fcd9c920aa089a2a8d75784a84d623" width="=240" height="240" align="center"/>
</td>
</tr></table>
<table><tr>   
<td><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/2c83e8ce81ee25becb185682c98ed31c00c67995" width="=240" height="240" align="center"/>
</td>
</tr></table>
This is called the dual problem. Since the dual maximization problem is a quadratic function of the C subject to linear constraints, it is efficiently solvable by quadratic programming algorithms.

Here, the variables C are defined such that
<table><tr>   
<td><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/cf0866d87cbe878e13e6a06560af15b9a9cc6bb0" width="=240" height="240" align="right"/>
</td>
</tr></table>


### Parameter Tuning


#### Kernel
For a dataset that is non-linearly separable, a way is to create nonlinear classifiers by applying the kernel trick to maximum-margin hyperplanes. The resulting algorithm is formally similar, except that every dot product is replaced by a nonlinear kernel function. This allows the algorithm to fit the maximum-margin hyperplane in a transformed feature space. The transformation may be nonlinear and the transformed space high-dimensional; although the classifier is a hyperplane in the transformed feature space, it may be nonlinear in the original input space.

Generally, Linear Kernel is the best choice as SVM is already a linear model and has the lowest computational complexity. More often, if the dataset is not linearly separable, Radial Basis Function is the next most common kernel to be used.

#### Regularization (C value)
The Regularization parameter (often termed as C parameter in python’s sklearn library) tells the SVM optimization how much you want to avoid misclassifying each training example.

For large values of C, the optimization will choose a smaller-margin hyperplane if that hyperplane does a better job of getting all the training points classified correctly. Conversely, a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane, even if that hyperplane misclassifies more points.             

<table><tr>
<td> <img src="https://miro.medium.com/max/600/1*1dwut8cWQ-39POHV48tv4w.png" width="940" height="940" align="left"//> </td>
<td> <img src="https://miro.medium.com/max/600/1*gt_dkcA5p0ZTHjIpq1qnLQ.png" width="940" height="940" align="right"/> </td>
</tr></table>
<b>Left: low regularization value, right: high regularization value</b>


The tradeoff is however, a larger value of C might result in overfitting of the model, which is undersirable as it does not generalise the model for other data sets. Consequently, a smaller value of C might result in too many misclassified data points, which means that the model is low in accuracy in the first place. Thus, choosing the right balance of the C value is important.

#### Gamma
The gamma parameter defines how far the influence of a single training example reaches, with low values meaning ‘far’ and high values meaning ‘close’. In other words, with low gamma, points far away from plausible separation line are considered in the calculation for the separation line. Where as high gamma means the points close to plausible line are considered in the calculation.          

<table><tr>
<td> <img src="https://miro.medium.com/max/600/1*dGDQxV8j83VB90skHsXktw.png" width="940" height="940" align="left"//> </td>
<td> <img src="https://miro.medium.com/max/600/1*ClmsnU_yb1YtIwAAr7krmg.png" width="940" height="940" align="right"/> </td>
</tr></table>

Similarly to regularization, there is a tradeoff between high and low values of Gamma. Higher values of Gamma may result in too strict rules and thus the model cannot find a linearly separable line. Whereas lower values of Gamma may result in including too much noise into the model by including further away points, which reduces the model accuracy. Thus, finding the right balance of Gamma is also important.

### Apply SVM model
For this dataset, we will perform SVM to the model and access its accuracy progressively, starting from no parameter tuning.

#### SVM without parameter tuning
First, we train our SVM model without parameter tuning. Note that the default kernel for sklearn svm is "rbf" and cost = 1.0 and gamma = 1/n where n is the number of samples.


```python
from sklearn import svm
#train svm model without standardization and no parameter tuning
clf_original = svm.SVC( probability = True, gamma = 'scale')
clf_original.fit(X_train, y_train)
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
        max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
        verbose=False)




```python
#plot roc for svm
get_roc(clf_original, y_test, X_test, "SVM default parameters")
print(classification_report(y_test, clf_original.predict(X_test)))
```

    Optimal Threshold: 0.16469105377809068
    


![png](defaults_files/defaults_111_1.png)


                  precision    recall  f1-score   support
    
               0       0.83      0.95      0.89      6762
               1       0.68      0.36      0.47      1987
    
        accuracy                           0.82      8749
       macro avg       0.76      0.66      0.68      8749
    weighted avg       0.80      0.82      0.79      8749
    
    


```python
#confusion matrix
confusion(y_test,clf_original.predict(X_test), "SVM default parameters")
```

    Of 1987 Defaulters, the SVM default parameters identified 713
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Actual</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>6432</td>
      <td>330</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1274</td>
      <td>713</td>
    </tr>
  </tbody>
</table>
</div>



Based on AUROC and Recall, the SVM model with default parameters seem to do average compared to the other models tested. Now let's search for the best parameters to tune the model.

#### SVM with Parameter tuning
One way to find the best parameters for the model is using grid search via GridSearchCV package from sklearn. 

Grid search is the process of performing hyper parameter tuning in order to determine the optimal values for a given model. This is significant as the performance of the entire model is based on the hyper parameter values specified.

GridSearchSV works by using a cross validation process to determine the hyper parameter value set which provides the best accuracy levels. We will start with the linear kernel and move on to rbf if necessary.


```python
from sklearn.model_selection import GridSearchCV
def svc_param_selection(X, y, nfolds):
    Cs = [0.01, 0.1, 1]
    gammas = [0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=nfolds, scoring = 'recall')
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_
svc_param_selection(X_train, y_train,2)

```




    {'C': 0.01, 'gamma': 0.01}



With 5 folds, it can be found that C = 0.01 , and gamma = 0.01 will have the best svm model with  kernel


```python
#train svm model with feature reduction and cost = 0.01, gamma = 0.01, linear kernel
clf_reduced_tuned = svm.SVC(kernel = 'linear', probability = True, C = 0.01, gamma = 0.01 )
clf_reduced_tuned.fit(X_train, y_train)
```




    SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=0.01, kernel='linear',
        max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
        verbose=False)




```python
auroc = get_roc(clf_reduced_tuned, y_test, X_test, 
        "SVM reduced features and tuning linear kernel")
print(classification_report(y_test, clf_reduced_tuned.predict(X_test)))
```

    Optimal Threshold: 0.15996357777982226
    


![png](defaults_files/defaults_118_1.png)


                  precision    recall  f1-score   support
    
               0       0.83      0.96      0.89      6762
               1       0.70      0.32      0.44      1987
    
        accuracy                           0.81      8749
       macro avg       0.77      0.64      0.66      8749
    weighted avg       0.80      0.81      0.79      8749
    
    


```python
#confusion matrix
confusion(y_test,clf_reduced_tuned.predict(X_test), "SVM reduced features and tuning linear kernal")
```

    Of 1987 Defaulters, the SVM reduced features and tuning linear kernal identified 638
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Actual</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>6492</td>
      <td>270</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1349</td>
      <td>638</td>
    </tr>
  </tbody>
</table>
</div>



As shown, the AUROC actually increased with tuning of parameters. Next, we will experiment with the RBF kernel


```python
#train svm model with feature reduction and cost = 0.1, gamma = 0.1, rbf kernel
clf_reduced_tuned_rbf = svm.SVC(kernel = 'rbf', probability = True, C = 0.1, gamma = 0.1)
clf_reduced_tuned_rbf.fit(X_train, y_train)
```




    SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
        max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
        verbose=False)




```python
auroc = get_roc(clf_reduced_tuned_rbf, y_test, X_test, 
        "SVM reduced features and tuning rbf kernel")
print(classification_report(y_test, clf_reduced_tuned_rbf.predict(X_test)))
```

    Optimal Threshold: 0.15910713557498266
    


![png](defaults_files/defaults_122_1.png)


                  precision    recall  f1-score   support
    
               0       0.84      0.95      0.89      6762
               1       0.67      0.38      0.48      1987
    
        accuracy                           0.82      8749
       macro avg       0.76      0.66      0.69      8749
    weighted avg       0.80      0.82      0.80      8749
    
    

As shown, the rbf kernel increases the AUROC and the recall increased to 0.40, thus, it can be said that the rbf kernel is better than the linear kernel. We will choose the RBF SVM as our best performing support vector machine.


```python
evaluation.loc[2] = (["SVM-RBF (Grid Search)" , 
                      classification_report(y_test, clf_reduced_tuned_rbf.predict(X_test), output_dict = True)["1"]["f1-score"],
                      auroc])

evaluation
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>F1-1</th>
      <th>AUROC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Decision Trees - Random Forest</td>
      <td>0.461339</td>
      <td>0.768458</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Logistic Regression (Optimal Cutoff)</td>
      <td>0.527665</td>
      <td>0.765244</td>
    </tr>
    <tr>
      <td>2</td>
      <td>SVM-RBF (Grid Search)</td>
      <td>0.482247</td>
      <td>0.748465</td>
    </tr>
  </tbody>
</table>
</div>



#### SVM with filtered features

We will now apply the best selected kernel (linear kernel) on filtered features to access AUROC and recall.


```python
clf_reduced_tuned_filtered = svm.SVC(kernel = 'rbf', probability = True)
clf_reduced_tuned_filtered.fit(X_train_filter, y_train)
```

    C:\Users\reonh\Anaconda3\lib\site-packages\sklearn\svm\base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
        kernel='rbf', max_iter=-1, probability=True, random_state=None,
        shrinking=True, tol=0.001, verbose=False)




```python
get_roc(clf_reduced_tuned_filtered, y_test, X_test_filter, 
        "SVM reduced features and tuning linear kernel")
```

    Optimal Threshold: 0.16104553371241384
    


![png](defaults_files/defaults_128_1.png)





    0.6689738476077944




```python
print(classification_report(y_test, clf_reduced_tuned_filtered.predict(X_test_filter)))
```

                  precision    recall  f1-score   support
    
               0       0.84      0.95      0.89      6762
               1       0.67      0.37      0.48      1987
    
        accuracy                           0.82      8749
       macro avg       0.75      0.66      0.68      8749
    weighted avg       0.80      0.82      0.79      8749
    
    

As we can see, the performance is not as great after using filtered features. The AUROC decreased while the recall remained the same. Thus, we will stick to using unfiltered features and SVM with rbf kernel.

### Neural Networks
We will now use the train and test sets as defined above and attempt to implement a neural network model on the data

#### Theory
A neural network is comprised of many layers of perceptrons that take in a vector as input and outputs a value. The outputs from one layer of perceptrons are passed into the next layer of perceptrons as input, until we reach the output layer. Each perceptron combines its input via an activation function. 

.


![image.png](https://www.researchgate.net/profile/Leslaw_Plonka/publication/260080460/figure/fig1/AS:340931325775876@1458295770470/A-simple-neural-network-diagram.png)


The network is at first randomly initialised with random weights on all its layers. Training samples are then passed into the network and predictions are made. The training error (difference between the actual value and the predicted value) is used to recalibrate the neural network by changing the weights. The change in weights is found via gradient descent, and  then backpropogated through the neural network to update all layers.


This process is repeated iteratively until the model converges (i.e. it cannot be improved further).

#### Training
Here we create an instance of our model, specifically a Sequential model, and add layers one at a time until we are happy with our network architecture. We will be training the model on our feature-selected dataset to speed up computation by reducing dimensionality. We have found that the performance difference between the 2 datasets are negligible.

We ensure the input layer has the right number of input features, and is specified when creating the first layer with the input_dim argument and setting it to 17 (The size of the feature selected dataset).
Additionaly, we start off using  a fully-connected network structure with three layers, and attempt to increase the number of layers at later part ater fully optimising our model.

Fully connected layers are defined using the Dense class. We  specify the number of neurons or nodes in the layer as the first argument, and specify the activation function using the activation argument. The rectified linear unit activation function (Relu) is usedon the first two layers and the Sigmoid function in the output layer.

Conventionally, Sigmoid and Tanh activation functions were preferred for all layers. However, better performance is achieved using the ReLU activation function. We use a sigmoid on the output layer to ensure our network output is between 0 and 1 and easy to map (binary) to either a probability of class 1 or snap to a hard classification of either class with a default threshold of 0.5.

The model expects rows of data with 17 variables (the input_dim=17 argument)
The first hidden layer has 17 nodes and uses the relu activation function.
The second hidden layer has 17 nodes and uses the relu activation function.
The output layer has one node and uses the sigmoid activation function.

#### Compiling

The model uses the efficient numerical libraries as the backend, and in this case Tensorflow is used. The backend automatically chooses the best way to represent the network for training and making predictions to run.

We must specify the loss function to use to evaluate a set of weights, the optimizer is used to search through different weights for the network and any optional metrics we would like to collect and report during training.

After experimenting with the various loss functions, such as hinge loss and binary cross entropy, we found that entropy performed the best for our dataset.

We have also found that among all the optimizers (Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax and Nadam) the optimizer "adam" is the most efficient yet yields the best results.

Additionaly, for this problem, we will run for a small number of epochs (20) and use a relatively small batch size of 10. This means that each epoch will involve (20/10) 2 updates to the model weights. After we have finalised the best optimizer, we will then increase the numebr of epochs to increase the number of updates to obtain a better result. 



```python
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# optimisers to try: Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax and Nadam
# verdict : Adam

# Loss function to try: Binary Cross Entropy, Hinge, Logcosh
# verdict: Binary Cross Entropy

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=17, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train_filter, y_train, epochs=20, batch_size=10)

```

    Using TensorFlow backend.
    

    Epoch 1/20
    17496/17496 [==============================] - 2s 110us/step - loss: 0.4708 - accuracy: 0.7956
    Epoch 2/20
    17496/17496 [==============================] - 2s 107us/step - loss: 0.4502 - accuracy: 0.8116
    Epoch 3/20
    17496/17496 [==============================] - 2s 113us/step - loss: 0.4477 - accuracy: 0.8125
    Epoch 4/20
    17496/17496 [==============================] - 2s 96us/step - loss: 0.4461 - accuracy: 0.8130
    Epoch 5/20
    17496/17496 [==============================] - 2s 105us/step - loss: 0.4450 - accuracy: 0.8133
    Epoch 6/20
    17496/17496 [==============================] - 2s 109us/step - loss: 0.4443 - accuracy: 0.81450s - loss: 0.4424 - 
    Epoch 7/20
    17496/17496 [==============================] - 2s 96us/step - loss: 0.4437 - accuracy: 0.8150
    Epoch 8/20
    17496/17496 [==============================] - 2s 101us/step - loss: 0.4435 - accuracy: 0.8144
    Epoch 9/20
    17496/17496 [==============================] - 2s 103us/step - loss: 0.4433 - accuracy: 0.8147
    Epoch 10/20
    17496/17496 [==============================] - 2s 101us/step - loss: 0.4429 - accuracy: 0.8142
    Epoch 11/20
    17496/17496 [==============================] - 2s 118us/step - loss: 0.4418 - accuracy: 0.8132
    Epoch 12/20
    17496/17496 [==============================] - 2s 112us/step - loss: 0.4416 - accuracy: 0.81450s - loss: 0.4413 - accuracy: 
    Epoch 13/20
    17496/17496 [==============================] - 2s 97us/step - loss: 0.4419 - accuracy: 0.8138
    Epoch 14/20
    17496/17496 [==============================] - 2s 128us/step - loss: 0.4417 - accuracy: 0.8140
    Epoch 15/20
    17496/17496 [==============================] - 2s 115us/step - loss: 0.4415 - accuracy: 0.8142
    Epoch 16/20
    17496/17496 [==============================] - 2s 118us/step - loss: 0.4415 - accuracy: 0.8141
    Epoch 17/20
    17496/17496 [==============================] - 2s 108us/step - loss: 0.4409 - accuracy: 0.8152
    Epoch 18/20
    17496/17496 [==============================] - 2s 127us/step - loss: 0.4413 - accuracy: 0.8145
    Epoch 19/20
    17496/17496 [==============================] - 2s 126us/step - loss: 0.4403 - accuracy: 0.8145
    Epoch 20/20
    17496/17496 [==============================] - 2s 110us/step - loss: 0.4405 - accuracy: 0.8156
    




    <keras.callbacks.callbacks.History at 0x17a5fc89b38>




```python
get_roc(model, y_test, X_test_filter,  "Neural Network 17x8x8x1 Adam, Entropy, 20 epoch")
predictions = list(model.predict(X_test_filter).ravel() > 0.5)
print(classification_report(y_test,predictions))
```

    Optimal Threshold: 0.23287344
    


![png](defaults_files/defaults_135_1.png)


                  precision    recall  f1-score   support
    
               0       0.84      0.94      0.89      6762
               1       0.65      0.39      0.49      1987
    
        accuracy                           0.81      8749
       macro avg       0.75      0.66      0.69      8749
    weighted avg       0.80      0.81      0.80      8749
    
    

Experimenting with lowering the cutoff for the neural network,


```python
optimal_adam = get_optimal(model, y_train, X_train_filter, "Neural Network 17x8x8x1 Adam Entropy")
auroc = get_roc(model, y_test, X_test_filter, "Neural Network 17x8x8x1 Adam, Entropy")
predictions = list(model.predict(X_test_filter).ravel() > optimal_adam)
print(classification_report(y_test,predictions))
```

    Optimal Threshold: 0.23287344
    


![png](defaults_files/defaults_137_1.png)


                  precision    recall  f1-score   support
    
               0       0.87      0.81      0.84      6762
               1       0.48      0.60      0.54      1987
    
        accuracy                           0.76      8749
       macro avg       0.68      0.71      0.69      8749
    weighted avg       0.79      0.76      0.77      8749
    
    

The performance is quite impressive when we lowered the classification cut off. The ROC of 0.76 is also on par with other models. Now we ramp up the number of epochs.


```python
model50 = Sequential()
model50.add(Dense(12, input_dim=17, activation='relu'))
model50.add(Dense(8, activation='relu'))
model50.add(Dense(8, activation='relu'))
model50.add(Dense(8, activation='relu'))
model50.add(Dense(1, activation='sigmoid'))
# compile the keras model
model50.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model50.fit(X_train_filter, y_train, epochs=50, batch_size=10)

```

    Epoch 1/50
    17496/17496 [==============================] - 2s 112us/step - loss: 0.4863 - accuracy: 0.7969
    Epoch 2/50
    17496/17496 [==============================] - 2s 102us/step - loss: 0.4511 - accuracy: 0.8122
    Epoch 3/50
    17496/17496 [==============================] - 2s 103us/step - loss: 0.4485 - accuracy: 0.8124
    Epoch 4/50
    17496/17496 [==============================] - 2s 105us/step - loss: 0.4472 - accuracy: 0.81200s - loss: 0.4465 
    Epoch 5/50
    17496/17496 [==============================] - 2s 103us/step - loss: 0.4461 - accuracy: 0.8123
    Epoch 6/50
    17496/17496 [==============================] - 2s 119us/step - loss: 0.4450 - accuracy: 0.8124
    Epoch 7/50
    17496/17496 [==============================] - 2s 112us/step - loss: 0.4432 - accuracy: 0.8138
    Epoch 8/50
    17496/17496 [==============================] - 2s 103us/step - loss: 0.4428 - accuracy: 0.8139
    Epoch 9/50
    17496/17496 [==============================] - 2s 104us/step - loss: 0.4422 - accuracy: 0.8132
    Epoch 10/50
    17496/17496 [==============================] - 2s 101us/step - loss: 0.4420 - accuracy: 0.8140
    Epoch 11/50
    17496/17496 [==============================] - 2s 97us/step - loss: 0.4411 - accuracy: 0.8137
    Epoch 12/50
    17496/17496 [==============================] - 2s 104us/step - loss: 0.4410 - accuracy: 0.8146
    Epoch 13/50
    17496/17496 [==============================] - 2s 97us/step - loss: 0.4412 - accuracy: 0.8143
    Epoch 14/50
    17496/17496 [==============================] - 2s 98us/step - loss: 0.4415 - accuracy: 0.8141
    Epoch 15/50
    17496/17496 [==============================] - 2s 120us/step - loss: 0.4402 - accuracy: 0.8149
    Epoch 16/50
    17496/17496 [==============================] - 2s 103us/step - loss: 0.4402 - accuracy: 0.8146
    Epoch 17/50
    17496/17496 [==============================] - 2s 100us/step - loss: 0.4406 - accuracy: 0.8141
    Epoch 18/50
    17496/17496 [==============================] - 2s 110us/step - loss: 0.4400 - accuracy: 0.8152
    Epoch 19/50
    17496/17496 [==============================] - 2s 102us/step - loss: 0.4397 - accuracy: 0.8140
    Epoch 20/50
    17496/17496 [==============================] - 2s 100us/step - loss: 0.4398 - accuracy: 0.8140
    Epoch 21/50
    17496/17496 [==============================] - 2s 104us/step - loss: 0.4396 - accuracy: 0.8153
    Epoch 22/50
    17496/17496 [==============================] - 2s 101us/step - loss: 0.4401 - accuracy: 0.8138
    Epoch 23/50
    17496/17496 [==============================] - 2s 112us/step - loss: 0.4394 - accuracy: 0.8150
    Epoch 24/50
    17496/17496 [==============================] - 2s 119us/step - loss: 0.4397 - accuracy: 0.8152
    Epoch 25/50
    17496/17496 [==============================] - 2s 117us/step - loss: 0.4396 - accuracy: 0.8148
    Epoch 26/50
    17496/17496 [==============================] - 2s 106us/step - loss: 0.4396 - accuracy: 0.8144
    Epoch 27/50
    17496/17496 [==============================] - 2s 95us/step - loss: 0.4393 - accuracy: 0.8135
    Epoch 28/50
    17496/17496 [==============================] - 2s 100us/step - loss: 0.4390 - accuracy: 0.8152
    Epoch 29/50
    17496/17496 [==============================] - 2s 115us/step - loss: 0.4392 - accuracy: 0.8138
    Epoch 30/50
    17496/17496 [==============================] - 2s 100us/step - loss: 0.4391 - accuracy: 0.8141
    Epoch 31/50
    17496/17496 [==============================] - 2s 102us/step - loss: 0.4397 - accuracy: 0.8149
    Epoch 32/50
    17496/17496 [==============================] - 2s 127us/step - loss: 0.4390 - accuracy: 0.8147
    Epoch 33/50
    17496/17496 [==============================] - 2s 115us/step - loss: 0.4388 - accuracy: 0.8145
    Epoch 34/50
    17496/17496 [==============================] - 2s 115us/step - loss: 0.4385 - accuracy: 0.8153
    Epoch 35/50
    17496/17496 [==============================] - 2s 102us/step - loss: 0.4389 - accuracy: 0.8150
    Epoch 36/50
    17496/17496 [==============================] - 2s 100us/step - loss: 0.4391 - accuracy: 0.8146
    Epoch 37/50
    17496/17496 [==============================] - 2s 107us/step - loss: 0.4386 - accuracy: 0.8142
    Epoch 38/50
    17496/17496 [==============================] - 2s 102us/step - loss: 0.4386 - accuracy: 0.8143
    Epoch 39/50
    17496/17496 [==============================] - 2s 101us/step - loss: 0.4379 - accuracy: 0.8148
    Epoch 40/50
    17496/17496 [==============================] - 2s 110us/step - loss: 0.4385 - accuracy: 0.8148
    Epoch 41/50
    17496/17496 [==============================] - 2s 116us/step - loss: 0.4386 - accuracy: 0.8149
    Epoch 42/50
    17496/17496 [==============================] - 2s 104us/step - loss: 0.4380 - accuracy: 0.8150
    Epoch 43/50
    17496/17496 [==============================] - 2s 108us/step - loss: 0.4382 - accuracy: 0.8144
    Epoch 44/50
    17496/17496 [==============================] - 2s 101us/step - loss: 0.4381 - accuracy: 0.8152
    Epoch 45/50
    17496/17496 [==============================] - 2s 105us/step - loss: 0.4378 - accuracy: 0.8151
    Epoch 46/50
    17496/17496 [==============================] - 2s 104us/step - loss: 0.4374 - accuracy: 0.8153
    Epoch 47/50
    17496/17496 [==============================] - 2s 99us/step - loss: 0.4382 - accuracy: 0.8152
    Epoch 48/50
    17496/17496 [==============================] - 2s 110us/step - loss: 0.4379 - accuracy: 0.8166
    Epoch 49/50
    17496/17496 [==============================] - 2s 112us/step - loss: 0.4381 - accuracy: 0.8144
    Epoch 50/50
    17496/17496 [==============================] - 2s 109us/step - loss: 0.4378 - accuracy: 0.8154
    




    <keras.callbacks.callbacks.History at 0x17a5fe12630>



We observe that the accuracy did not increase much at all from the 20th to 50th epoch. In such a situation we will choose the 20 epoch model for its faster computation.


```python
optimal_adam50 = get_optimal(model50, y_train, X_train_filter, "Neural Network 17x8x8x1 Adam Entropy")
get_roc(model50, y_test, X_test_filter, "Neural Network 17x8x8x1 Adam, Entropy, 50 epoch")
predictions50 = list(model50.predict(X_test_filter).ravel() > optimal_adam50)
print(classification_report(y_test,predictions50))
```

    Optimal Threshold: 0.20843479
    


![png](defaults_files/defaults_141_1.png)


                  precision    recall  f1-score   support
    
               0       0.87      0.82      0.84      6762
               1       0.48      0.59      0.53      1987
    
        accuracy                           0.76      8749
       macro avg       0.68      0.70      0.69      8749
    weighted avg       0.78      0.76      0.77      8749
    
    


```python
evaluation.loc[3] = (["Neural Network - 17x8x8x1 Adam, Entropy, 20 Epochs" , 
                      classification_report(y_test, predictions, output_dict = True)["1"]["f1-score"],
                      auroc])

evaluation
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>F1-1</th>
      <th>AUROC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Decision Trees - Random Forest</td>
      <td>0.461339</td>
      <td>0.768458</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Logistic Regression (Optimal Cutoff)</td>
      <td>0.527665</td>
      <td>0.765244</td>
    </tr>
    <tr>
      <td>2</td>
      <td>SVM-RBF (Grid Search)</td>
      <td>0.482247</td>
      <td>0.748465</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Neural Network - 17x8x8x1 Adam, Entropy, 20 Ep...</td>
      <td>0.535834</td>
      <td>0.741382</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Naive Bayes</td>
      <td>0.526342</td>
      <td>0.741382</td>
    </tr>
  </tbody>
</table>
</div>



### Naive Bayes
#### Theory
Naive Bayes classifier is a probabilistic machine learning model used for classification. The crux of the classifier is based on the Bayes theorem.
##### Bayes Theorem:
![image.png](https://miro.medium.com/max/510/1*tjcmj9cDQ-rHXAtxCu5bRQ.png)
Using the Bayes theorem, we can find the probability of A happening, given that B has occured.
One assumption about naive bayes is that the predictors/features are independent.

#### Training the Naive bayes model


```python
from sklearn import datasets
from sklearn import metrics
import matplotlib.pyplot as plt
import time
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

gnb = GaussianNB()
```


```python
#training naive bayes model
gnb.fit(X_train, y_train)
```




    GaussianNB(priors=None, var_smoothing=1e-09)




```python
#classifying values
predicted = gnb.predict(X_train)
expected = y_train
```


```python
#plot roc for naive Bayes
auroc = get_roc(gnb, y_test, X_test, "Naive Bayes")
```

    Optimal Threshold: 0.9999935527715175
    


![png](defaults_files/defaults_148_1.png)



```python
#accessing model performance
#print(metrics.confusion_matrix(expected, predicted))
print(classification_report(y_train,gnb.predict(X_train)))
```

                  precision    recall  f1-score   support
    
               0       0.91      0.11      0.20     13442
               1       0.25      0.96      0.39      4054
    
        accuracy                           0.31     17496
       macro avg       0.58      0.54      0.29     17496
    weighted avg       0.76      0.31      0.24     17496
    
    

We observe that while the recall is 0.96, the f1 is 0.39 and the overall accuracy is atrocious. 

We will now try searching for the smoothing parameter to achieve a better ROC and F1 on default. After some experimentation we found that 0.01 is a good value for this parameter.


```python
gnb = GaussianNB(var_smoothing = 0.01)
#experimenting with smoothing constant of naive bayes model on the training set.
gnb.fit(X_train, y_train)
auroc = get_roc(gnb, y_test, X_test, "Naive Bayes")
print(classification_report(y_train,gnb.predict(X_train)))
```

    Optimal Threshold: 0.038218795916133315
    


![png](defaults_files/defaults_151_1.png)


                  precision    recall  f1-score   support
    
               0       0.86      0.85      0.85     13442
               1       0.52      0.52      0.52      4054
    
        accuracy                           0.78     17496
       macro avg       0.69      0.69      0.69     17496
    weighted avg       0.78      0.78      0.78     17496
    
    

Smoothing constant of 0.01 allowed us to acheieve a recall and f1 of 0.52 on the training set. Applied on the test set:


```python
#plot roc for naive Bayes
auroc = get_roc(gnb, y_test, X_test, "Naive Bayes")
print(classification_report(y_test,gnb.predict(X_test)))
```

    Optimal Threshold: 0.038218795916133315
    


![png](defaults_files/defaults_153_1.png)


                  precision    recall  f1-score   support
    
               0       0.86      0.86      0.86      6762
               1       0.52      0.53      0.53      1987
    
        accuracy                           0.78      8749
       macro avg       0.69      0.69      0.69      8749
    weighted avg       0.78      0.78      0.78      8749
    
    


```python
evaluation.loc[5] = (["Naive Bayes" , 
                      classification_report(y_test, gnb.predict(X_test), output_dict = True)["1"]["f1-score"],
                      auroc])

```


```python
evaluation.sort_values(["AUROC"], ascending = False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>F1-1</th>
      <th>AUROC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Decision Trees - Random Forest</td>
      <td>0.461339</td>
      <td>0.768458</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Logistic Regression (Optimal Cutoff)</td>
      <td>0.527665</td>
      <td>0.765244</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Neural Network 17x8x8x1</td>
      <td>0.535834</td>
      <td>0.761854</td>
    </tr>
    <tr>
      <td>2</td>
      <td>SVM-RBF (Grid Search)</td>
      <td>0.482247</td>
      <td>0.748465</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Naive Bayes</td>
      <td>0.526342</td>
      <td>0.741382</td>
    </tr>
  </tbody>
</table>
</div>




```python
raise(Exception("Stop Running"))
```


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    <ipython-input-106-c8b15946227a> in <module>
    ----> 1 raise(Exception("Stop Running"))
    

    Exception: Stop Running


## Appendix: Tuning Neural Network with different optimisers 
### Adamax Optimizer


```python
model = Sequential()
model.add(Dense(12, input_dim=46, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=10, batch_size=10)
predictions = list(model.predict(X_test).ravel() > 0.5)
#confusion(y_test, model.predict(X_test[sig.index]), "Deep Learning Keras Model")
auroc = get_roc(model, y_test, X_test, "Tuning Keras Model")

print(classification_report(y_test,predictions))
```

### Adadelta Optimizer


```python
model = Sequential()
model.add(Dense(12, input_dim=46, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=10, batch_size=10)
predictions = list(model.predict(X_test).ravel() > 0.5)
#confusion(y_test, model.predict(X_test[sig.index]), "Deep Learning Keras Model")
auroc = get_roc(model, y_test, X_test, "Tuning Keras Model")

print(classification_report(y_test,predictions))
```

### Adagrad Optimzier


```python
model = Sequential()
model.add(Dense(12, input_dim=46, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=10, batch_size=10)
predictions = list(model.predict(X_test).ravel() > 0.5)
#confusion(y_test, model.predict(X_test[sig.index]), "Deep Learning Keras Model")
auroc = get_roc(model, y_test, X_test, "Tuning Keras Model")

print(classification_report(y_test,predictions))


```

### RMSProp


```python
model = Sequential()
model.add(Dense(12, input_dim=46, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=10, batch_size=10)
predictions = list(model.predict(X_test).ravel() > 0.5)
#confusion(y_test, model.predict(X_test[sig.index]), "Deep Learning Keras Model")
auroc = get_roc(model, y_test, X_test, "Tuning Keras Model")

print(classification_report(y_test,predictions))
```

### SGD


```python
model = Sequential()
model.add(Dense(12, input_dim=46, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=10, batch_size=10)
predictions = list(model.predict(X_test).ravel() > 0.5)
#confusion(y_test, model.predict(X_test[sig.index]), "Deep Learning Keras Model")
auroc = get_roc(model, y_test, X_test, "Tuning Keras Model")

print(classification_report(y_test,predictions))
```

#### We conclude that best optimizer is adagrad. Testing it on the test set.


```python
model = Sequential()
model.add(Dense(12, input_dim=17, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train_filter, y_train, epochs=10, batch_size=10)
predictions = list(model.predict(X_test_filter).ravel() > 0.5)
#confusion(y_test, model.predict(X_test[sig.index]), "Deep Learning Keras Model")
auroc = get_roc(model, y_test, X_test_filter, "Neural Network - adagrad (filtered features)")

print(classification_report(y_test,predictions))

evaluation.loc[6] = (["Neural Network - adagrad" , 
                      classification_report(y_test, predictions, output_dict = True)["1"]["recall"],
                      auroc])

evaluation
```


```python
model = Sequential()
model.add(Dense(12, input_dim=46, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, y_train, epochs=10, batch_size=10)
predictions = list(model.predict(X_test).ravel() > 0.5)
#confusion(y_test, model.predict(X_test[sig.index]), "Deep Learning Keras Model")
auroc = get_roc(model, y_test, X_test, "Neural Network - adagrad (all features)")

print(classification_report(y_test,predictions))


evaluation.loc[6] = (["Neural Network - adagrad" , 
                      classification_report(y_test, predictions, output_dict = True)["1"]["recall"],
                      auroc])

evaluation

```
