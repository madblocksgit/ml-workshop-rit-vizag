
# 0-7893015625
# Madhu Parvathaneni

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from flask import Flask, render_template, request

dataset=pd.read_csv('Dataset_spine.csv')
#print(dataset)

# Seperate the Input and Output Variables
# Independent Variable alias Input Variable
X=dataset.iloc[:,0:12].values
#print(X)
# Dependent Variable alias Output Variable
Y=dataset.iloc[:,12].values

dummy=[]
for i in Y:
	if i=='Abnormal':
		dummy.append(0)
	elif i=='Normal':
		dummy.append(1)
	else:
		dummy.append(-1)
#print(dummy)
Y=dummy

# Spliting the dataset into train and test
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=1)

print(X_train.shape)
print(X_test.shape)

print(len(Y_train))
print(len(Y_test))

'''classifier=KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
print(accuracy_score(Y_pred,Y_test))'''

classifier=LogisticRegression()
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
print(accuracy_score(Y_pred,Y_test))

app=Flask(__name__) # Flask App Name

# API Routing

@app.route('/')
def get_connected():
	return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def read_data():
	f1=float(request.form['f1'])
	f2=float(request.form['f2'])
	f3=float(request.form['f3'])
	f4=float(request.form['f4'])
	f5=float(request.form['f5'])
	f6=float(request.form['f6'])
	f7=float(request.form['f7'])
	f8=float(request.form['f8'])
	f9=float(request.form['f9'])
	f10=float(request.form['f10'])
	f11=float(request.form['f11'])
	f12=float(request.form['f12'])

	print(f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12)
	dummy=classifier.predict([[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12]])
	print(dummy)
	out=''
	if(dummy[0]==0):
		out='Abnormal'
	elif(dummy[0]==1):
		out='Normal'
	return render_template('index.html',predicted_output=out)

if __name__=="__main__":
	app.run(debug=True) # Web Server


