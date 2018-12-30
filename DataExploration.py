#Loading libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline'
#plt.rcParams['figure.figsize'] = (10.0, 8.0)
x = np.linspace(0, 6, 200)
y = np.sin(x)
# Set figures to be large
plt.rcParams['figure.figsize'] = (10, 8)
plt.plot(x, y)

import seaborn as sns
from scipy import stats
from scipy.stats import norm

#loading data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
train.head()
print ('The train data has {0} rows and {1} columns'.format(train.shape[0],train.shape[1]))
print ('----------------------------')
print ('The test data has {0} rows and {1} columns'.format(test.shape[0],test.shape[1]))
train.info()
train.columns[train.isnull().any()]
miss = train.isnull().sum()/len(train)
miss = miss[miss > 0]
miss.sort_values(inplace=True)
print(miss)

#visualising missing values
miss = miss.to_frame()
miss.columns = ['count']
miss.index.names = ['Name']
miss['Name'] = miss.index

#plot the missing value count
sns.set(style="whitegrid", color_codes=True)
sns.barplot(x = 'Name', y = 'count', data=miss)
plt.xticks(rotation = 90)
plt.show()

#SalePrice
sns.distplot(train['SalePrice'])
plt.show()

#skewness
print ("The skewness of SalePrice is {}".format(train['SalePrice'].skew()))

#now transforming the target variable
target = np.log(train['SalePrice'])
print ('Skewness is', target.skew())
sns.distplot(target)
plt.show()

#separate variables into new data frames
numeric_data = train.select_dtypes(include=[np.number])
cat_data = train.select_dtypes(exclude=[np.number])
print ("There are {} numeric and {} categorical columns in train data".format(numeric_data.shape[1],cat_data.shape[1]))

del numeric_data['Id']

#correlation plot
corr = numeric_data.corr()
sns.heatmap(corr)
plt.show()

print (corr['SalePrice'].sort_values(ascending=False)[:15], '\n') #top 15 values
print ('----------------------')
print (corr['SalePrice'].sort_values(ascending=False)[-5:]) #last 5 values`

train['OverallQual'].unique()

#let's check the mean price per quality and plot it.
pivot = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
pivot.sort_values

pivot.plot(kind='bar', color='red')
plt.show()

#GrLivArea variable
sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'])
plt.show()

cat_data.describe()

sp_pivot = train.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
sp_pivot

sp_pivot.plot(kind='bar',color='red')
plt.show()





cat_data['SalePrice'] = train.SalePrice.values
k = anova(cat_data) 
k['disparity'] = np.log(1./k['pval'].values) 
sns.barplot(data=k, x = 'features', y='disparity') 
plt.xticks(rotation=90) 
plt.show()


#create numeric plots
num = [f for f in train.columns if train.dtypes[f] != 'object']
num.remove('Id')
nd = pd.melt(train, value_vars = num)
n1 = sns.FacetGrid (nd, col='variable', col_wrap=4, sharex=False, sharey = False)
n1 = n1.map(sns.distplot, 'value')
plt.show()


def boxplot(x,y,**kwargs):
            sns.boxplot(x=x,y=y)
            x = plt.xticks(rotation=90)

cat = [f for f in train.columns if train.dtypes[f] == 'object']

p = pd.melt(train, id_vars='SalePrice', value_vars=cat)
g = sns.FacetGrid (p, col='variable', col_wrap=2, sharex=False, sharey=False, size=5)
g = g.map(boxplot, 'value','SalePrice')
plt.show()



##################################   Data Pre-Processing     ######################################
