import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('muct76-opencv.csv')
df.drop('tag',axis=1,inplace=True)
df['A'], df['B'] = df['name'].str.split('-', 1).str
df.drop('A',axis=1,inplace=True)
df.drop('name',axis=1,inplace=True)
a=[]
for i in df['B']:
    s=list(i)
    a.append(s[0])
df.drop('B',axis=1,inplace=True)
df['Gender']=a

for i in df.index:
  if (df.loc[i,'Gender']=='m'):
   df.loc[i,'Gender']=1.0
  else:
     df.loc[i,'Gender']=0.0


zx=np.load('ac.npy')
target=df['Gender']
target=list(target)
df.drop('Gender',axis=1,inplace=True)
##from sklearn.ensemble import RandomForestClassifier
##from sklearn.feature_selection import SelectFromModel
##clf=RandomForestClassifier(n_estimators=50,max_features='sqrt')
##clf.fit(df,target)
##model=SelectFromModel(clf,prefit=True)
##train_reduced=model.transform(df)
##zc=model.transform(zx)
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
##print (train_reduced[1]).shape
train_red=pca.fit_transform(df)
za=pca.transform(zx)
print(za.shape)
za.reshape(-1,za.shape[1])
print(za.shape)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model=LinearDiscriminantAnalysis()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_red, target, test_size=0.2)
print type(X_test[1])
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
print (model.predict(za))
