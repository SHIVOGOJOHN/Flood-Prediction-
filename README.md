#import required dependancies 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as ts 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression, ElasticNet 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.svm import SVR 
from sklearn.neural_network import MLPRegressor 
from sklearn.metrics import mean_squared_error
 
#Load, read and clean data/Descriptive analysis 
df=pd.read_csv("flood.csv")
print(df.isnull().sum())

print(df.head())
print(df.describe())
print(df.info())

#Exploratory Data Analysis 
import matplotlib.pyplot as plt
import seaborn as sb
#correlation  using a heatmap
plt.figure()
sb.heatmap(df.corr()>0.7, annot=True,cbar=False)
plt.show()

#Histogram to show data distribution
df.hist(bins=21,figsize=(10,10))
plt.show()



#Identify feature and target values 
x=df.drop(["FloodProbability"], axis=1)
y=df["FloodProbability"]

#Model selection 
xtrain,xtest,ytrain, ytest = ts(x,y,test_size=0.3, random_state=20)

#Normalize the data
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)

#Fit the models 
lr=LinearRegression()
lr.fit(xtrain,ytrain)
predictlr=lr.predict(xtest)

sv=SVR()
sv.fit(xtrain, ytrain )
predictsv=sv.predict(xtest)

rf=RandomForestRegressor()
rf.fit(xtrain, ytrain)
predictrf=rf.predict(xtest)

mlp=MLPRegressor(hidden_layer_sizes=(11,11,11), max_iter=300)  
mlp.fit(xtrain, ytrain)
predictmlp=mlp.predict(xtest)

en=ElasticNet()
en.fit(xtrain, ytrain)
predicten=en.predict(xtest)


#Evaluate performance of each model
print(mean_squared_error(ytest, predictlr))
print(mean_squared_error(ytest, predictsv))
print(mean_squared_error(ytest, predictrf))
print(mean_squared_error(ytest, predictmlp))
print(mean_squared_error(ytest, predicten))


#Carry out Predictions from the data provided
#use RandomForestRegressor as it has the lowest mean squared error
new_data=pd.read_csv("flood.csv2 (1).csv")#dataset for Prediction 
new_data=sc.transform(new_data)
prediction=rf.predict(new_data)
print(prediction)


#merge the data used as well as the Prediction results into a new file.
prediction=np.array([0.48115,0.51045, 0.4667,  0.51625,0.5123, 0.4744, 0.4754,  0.51015, 0.50665, 0.5039 ]).reshape(10,1)

p_new=pd.DataFrame(prediction)
n_new=pd.DataFrame(new_data)

final_data=pd.concat([n_new, p_new],axis=1)

final_data.to_csv("flood (2).csv")
