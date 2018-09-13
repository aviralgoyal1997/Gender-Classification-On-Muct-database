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
from sklearn.ensemble import RandomForestClassifier
#this is a comment
##from sklearn.feature_selection import SelectFromModel
##clf=RandomForestClassifier(n_estimators=50,max_features='sqrt')
##clf.fit(df,target)
##model=SelectFromModel(clf,prefit=True)
##train_reduced=model.transform(df)
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
train_red=pca.fit_transform(df)
za=pca.transform(zx)
parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}

model=RandomForestClassifier(**parameters)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_red, target, test_size=0.2)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
print (model.predict(za))
