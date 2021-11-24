import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
matplotlib.rcParams["figure.figsize"] = (20,10)

df1 = pd.read_csv('H:/Project/Bengaluru_House_Data.csv')

print(df1)

print(df1.head())
print("")

#print rows and column
print(df1.shape)
print("")

#pie chart for area type
plt.pie(df1['area_type'].value_counts().values,labels = df1['area_type'].value_counts().index, explode=(0.1,0,0,0.1),shadow=True,autopct='%1.1f%%') 
plt.title("Area Type")
plt.show()

#removing unimportant column(cleaning datataset)
df1.groupby('area_type')['area_type'].count()
df2 = df1.drop(['area_type', 'availability', 'society', 'balcony'], axis = 'columns')
print(df2.head())

df3 = df2.dropna()
print(df3.isnull().sum())
print("")

#unique values of size of house
print(df3['size'].unique())

#creating a column bhk storing no. of BHK or Bedroom
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(" ")[0]))
print(df3.head())
print("")

#checking unique values in bhk column
print(df3['bhk'].unique())
print("")

#checking size of bhk greater than 20 
print(df3[df3.bhk>20])
print("")

#checking unique value of total_sqft
print(df3.total_sqft.unique())
print("")

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

#checking values of total_sqft values which is not float
print(df3[~df3['total_sqft'].apply(is_float)].head(10))
print("")

#converting into single float value
def convert_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_to_num)
print(df4.head(3))
print("")

print(df4.loc[30])
print("")

df4_1=df4.head(100)

#graph for first 100 data
x1=df4_1.bhk
y1=df4_1.price
x2=df4_1.total_sqft


plt.subplot(121)
plt.scatter(x1,y1,c='b',s=50,marker=".",label="size vs price")
plt.title("Size vs Price")
plt.xlabel("Size(BHK)")
plt.ylabel("Price(in lakh)")

plt.subplot(122)
plt.scatter(x2,y1,c='g',s=50,marker=".",label="Square_feet vs price")
plt.title("Square_feet vs price")
plt.xlabel("Square_feet")
plt.ylabel("Price(in lakh)")

plt.show()

#histrogram for first 100 data
plt.hist(df4_1.total_sqft,histtype='bar',rwidth=0.8)
plt.title("No. of counts of Square Feet")
plt.xlabel("Square_feet")
plt.ylabel("COUNT")
plt.show()

plt.hist(df4_1.bath,rwidth=0.8)
plt.title("No. of counts of Bathroom")
plt.xlabel("NO.OF Bathroom")
plt.ylabel("COUNT")
plt.show()

#heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df4.corr(),annot=True)
plt.title("Heatmap of Bengaluru House Pricing data")
plt.show()

#calculating price_per_sqft for calculation
df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
print(df5.head())
print("")

#checking no. of unique location
print(len(df5.location.unique()))
print("")

df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].count().sort_values(ascending = False)

print(len(location_stats[location_stats<=10]))
print("")

location_stats_lessthan_10 = location_stats[location_stats<=10]

df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_lessthan_10 else x)
print(len(df5.location.unique()))
print("")


df6 = df5[~(df5.total_sqft/df5.bhk<300)]
print(df6.shape)

def remove_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index = True)
    return df_out

df7 = remove_outliers(df6)
print(df7.shape)

#

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)

df9 = df8[df8.bath < df8.bhk+2]

df10 = df9.drop(['size','price_per_sqft'],axis='columns')
print(df10.head(3))

#converting location datatype string to numerical
dummies = pd.get_dummies(df10.location)

#combing new columns of locations
df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')

#removing location column
df12 = df11.drop('location',axis='columns')
print(df12.head(2))

#value of X & y
X = df12.drop(['price'],axis='columns')
y = df12.price


#taking train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

#linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
#acurracy
print("")
print("Linear Regressor")
print('Test score:',lr.score(X_test,y_test))
y_pred=lr.predict(X_test)
from sklearn.metrics import mean_squared_error
print('Mean squared error:',mean_squared_error(y_test,y_pred))



# decision tree
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(X_train,y_train)
print("")
print("Decision Tree Regressor")
print('Test score:',dt.score(X_test,y_test))
y_predd=dt.predict(X_test)
print('Mean squared error:',mean_squared_error(y_test,y_predd))



#random forest
from sklearn.ensemble import RandomForestRegressor
Rdrm = RandomForestRegressor() 
Rdrm.fit(X_train, y_train)
print("")
print("Random Forest Regressor")
print('Test score:',Rdrm.score(X_test, y_test)) 
y_predr=Rdrm.predict(X_test)
print('Mean squared error:',mean_squared_error(y_test,y_predr))
print("")






#Price_Predict function
def price_predict(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr.predict([x])[0]

#prediction
print("  TEST CASES  ")
#Test Case 1:-
print('1st Case:',price_predict('1st Phase JP Nagar',1000, 2, 2))
#Test Case 2:-
print('2nd Case:',price_predict('Indira Nagar',2000, 3, 3))
#Test Case 3:-
print('3rd Case:',price_predict('Uttarahalli',1500, 2, 2))
#Test Case 4:-
print('4th Case:',price_predict('Kothanur',1000, 1, 1))
#Test Case 5:-
print('5th Case:',price_predict('Electronic City Phase II',2500, 2, 4))
#Test Case 6:-
print('6th Case:',price_predict('Yeshwanthpur',2200, 2, 3))
#Test Case 7:-
print('7th Case:',price_predict('1st Block Jayanagar',4000, 4, 6))
#Test Case 8:-
print('8th Case:',price_predict('Yelenahalli',3000, 3, 2))
#Test Case 9:-
print('9th Case:',price_predict('KR Puram',3600, 2, 4))
#Test Case 10:-
print('10th Case:',price_predict('Hebbal',5132, 5, 7))

