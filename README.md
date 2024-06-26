# DiabetesPrediction

#### welcome to

# Diabetes EDA and ML prediction model

### this is an EDA on Diagnostic measurements of a specific demographic (pima indian heritage, Females,at least 21 years old) and its relation to whether they have Diabetes or not. Data is collected by National Institute of Diabetes and Kidney diseases. Dataset is available on [kaggle](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)


### Import libraries


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sklearn
```


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

### Import data from CSV file


```python
data = pd.read_csv(r"C:\Users\Administrator\Documents\safaa\meriskill\diabetes\diabetes.csv")
data
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>763</th>
      <td>10</td>
      <td>101</td>
      <td>76</td>
      <td>48</td>
      <td>180</td>
      <td>32.9</td>
      <td>0.171</td>
      <td>63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>764</th>
      <td>2</td>
      <td>122</td>
      <td>70</td>
      <td>27</td>
      <td>0</td>
      <td>36.8</td>
      <td>0.340</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>765</th>
      <td>5</td>
      <td>121</td>
      <td>72</td>
      <td>23</td>
      <td>112</td>
      <td>26.2</td>
      <td>0.245</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>766</th>
      <td>1</td>
      <td>126</td>
      <td>60</td>
      <td>0</td>
      <td>0</td>
      <td>30.1</td>
      <td>0.349</td>
      <td>47</td>
      <td>1</td>
    </tr>
    <tr>
      <th>767</th>
      <td>1</td>
      <td>93</td>
      <td>70</td>
      <td>31</td>
      <td>0</td>
      <td>30.4</td>
      <td>0.315</td>
      <td>23</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>768 rows × 9 columns</p>
</div>



### Explore Data


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Pregnancies               768 non-null    int64  
     1   Glucose                   768 non-null    int64  
     2   BloodPressure             768 non-null    int64  
     3   SkinThickness             768 non-null    int64  
     4   Insulin                   768 non-null    int64  
     5   BMI                       768 non-null    float64
     6   DiabetesPedigreeFunction  768 non-null    float64
     7   Age                       768 non-null    int64  
     8   Outcome                   768 non-null    int64  
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB
    


```python
data.describe()
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>120.894531</td>
      <td>69.105469</td>
      <td>20.536458</td>
      <td>79.799479</td>
      <td>31.992578</td>
      <td>0.471876</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>31.972618</td>
      <td>19.355807</td>
      <td>15.952218</td>
      <td>115.244002</td>
      <td>7.884160</td>
      <td>0.331329</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.078000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>62.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.300000</td>
      <td>0.243750</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>23.000000</td>
      <td>30.500000</td>
      <td>32.000000</td>
      <td>0.372500</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>140.250000</td>
      <td>80.000000</td>
      <td>32.000000</td>
      <td>127.250000</td>
      <td>36.600000</td>
      <td>0.626250</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### Check for missing Values


```python
data.isnull().sum()
```




    Pregnancies                 0
    Glucose                     0
    BloodPressure               0
    SkinThickness               0
    Insulin                     0
    BMI                         0
    DiabetesPedigreeFunction    0
    Age                         0
    Outcome                     0
    dtype: int64




```python
sns.heatmap(data.isnull())
```




    <Axes: >




    
![png](readme/output_12_1.png)
    



```python
data.nunique()
```




    Pregnancies                  17
    Glucose                     136
    BloodPressure                47
    SkinThickness                51
    Insulin                     186
    BMI                         248
    DiabetesPedigreeFunction    517
    Age                          52
    Outcome                       2
    dtype: int64



#### check for correlation matrix


```python
correlation = data.corr()
print(correlation)
```

                              Pregnancies   Glucose  BloodPressure  SkinThickness  \
    Pregnancies                  1.000000  0.129459       0.141282      -0.081672   
    Glucose                      0.129459  1.000000       0.152590       0.057328   
    BloodPressure                0.141282  0.152590       1.000000       0.207371   
    SkinThickness               -0.081672  0.057328       0.207371       1.000000   
    Insulin                     -0.073535  0.331357       0.088933       0.436783   
    BMI                          0.017683  0.221071       0.281805       0.392573   
    DiabetesPedigreeFunction    -0.033523  0.137337       0.041265       0.183928   
    Age                          0.544341  0.263514       0.239528      -0.113970   
    Outcome                      0.221898  0.466581       0.065068       0.074752   
    
                               Insulin       BMI  DiabetesPedigreeFunction  \
    Pregnancies              -0.073535  0.017683                 -0.033523   
    Glucose                   0.331357  0.221071                  0.137337   
    BloodPressure             0.088933  0.281805                  0.041265   
    SkinThickness             0.436783  0.392573                  0.183928   
    Insulin                   1.000000  0.197859                  0.185071   
    BMI                       0.197859  1.000000                  0.140647   
    DiabetesPedigreeFunction  0.185071  0.140647                  1.000000   
    Age                      -0.042163  0.036242                  0.033561   
    Outcome                   0.130548  0.292695                  0.173844   
    
                                   Age   Outcome  
    Pregnancies               0.544341  0.221898  
    Glucose                   0.263514  0.466581  
    BloodPressure             0.239528  0.065068  
    SkinThickness            -0.113970  0.074752  
    Insulin                  -0.042163  0.130548  
    BMI                       0.036242  0.292695  
    DiabetesPedigreeFunction  0.033561  0.173844  
    Age                       1.000000  0.238356  
    Outcome                   0.238356  1.000000  
    


```python
data.columns
```




    Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
          dtype='object')




```python
data.dtypes
```




    Pregnancies                   int64
    Glucose                       int64
    BloodPressure                 int64
    SkinThickness                 int64
    Insulin                       int64
    BMI                         float64
    DiabetesPedigreeFunction    float64
    Age                           int64
    Outcome                       int64
    dtype: object



### Visualize the Data


```python
print(plt.style.available)
plt.style.use('fivethirtyeight')
```

    ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']
    


```python
sns.heatmap(data.corr(numeric_only=True),annot = True)

plt.rcParams['figure.figsize']=(20,7)
plt.show()
```


    
![png](readme/output_20_0.png)
    



```python
data.plot()
```




    <Axes: >




    
![png](readme/output_21_1.png)
    



```python
data.plot(kind = 'line', subplots = True)
```




    array([<Axes: >, <Axes: >, <Axes: >, <Axes: >, <Axes: >, <Axes: >,
           <Axes: >, <Axes: >, <Axes: >], dtype=object)




    
![png](readme/output_22_1.png)
    



```python
data.boxplot(figsize=(20,15))
```




    <Axes: >




    
![png](readme/output_23_1.png)
    



```python
data.hist(figsize=(20,15))
```




    array([[<Axes: title={'center': 'Pregnancies'}>,
            <Axes: title={'center': 'Glucose'}>,
            <Axes: title={'center': 'BloodPressure'}>],
           [<Axes: title={'center': 'SkinThickness'}>,
            <Axes: title={'center': 'Insulin'}>,
            <Axes: title={'center': 'BMI'}>],
           [<Axes: title={'center': 'DiabetesPedigreeFunction'}>,
            <Axes: title={'center': 'Age'}>,
            <Axes: title={'center': 'Outcome'}>]], dtype=object)




    
![png](readme/output_24_1.png)
    


 

### Training The ML Model

#### using the data to successfully predict if the Patient has Diabetes or not based on the Diagnostic Variables

### Train test split


```python
X =data.drop("Outcome", axis=1)
Y =data['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
```


```python
model=LogisticRegression()
model.fit(X_train, Y_train)
```

 



```python
prediction= model.predict(X_test)
print(prediction)
```

    [0 1 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 1 1 0 1 0 0 1 1 0 0 0 0 0 0 0
     0 1 0 1 0 0 1 0 1 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
     0 1 0 0 1 0 1 0 0 0 0 0 0 1 1 0 1 0 0 0 1 1 0 1 1 0 0 0 0 0 1 0 0 0 1 0 0
     0 0 1 0 0 0 0 1 0 0 1 0 0 1 1 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 0 0 0 0
     1 0 0 1 1 0]
    

#### Accuracy


```python
accuracy= accuracy_score(prediction, Y_test)
print(accuracy)
```

    0.7272727272727273
    


```python
print("Accuracy on training set:", model.score(X_test, Y_test))
```

    Accuracy on training set: 0.7272727272727273
    
