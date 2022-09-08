from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt
import pandas_profiling as pp
from pandas_profiling import ProfileReport
import keras
import keras as k
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Activation
from keras import optimizers
from numpy.random import seed
seed(1)
df_users = pd.read_csv("users.csv")
df_fusers = pd.read_csv("fusers.csv")
isNotFake = np.zeros(1481)
isFake = np.ones(1337)
df_fusers["isFake"] = isFake
df_users["isFake"] = isNotFake
df_allUsers = pd.concat([df_fusers, df_users], ignore_index=True)
df_allUsers.columns = df_allUsers.columns.str.strip()
df_allUsers = df_allUsers.sample(frac=1).reset_index(drop=True)
Y = df_allUsers.isFake
df_allUsers.drop(["isFake"], axis=1, inplace=True)
X = df_allUsers
Y.reset_index(drop=True, inplace=True)
lang_list = list(enumerate(np.unique(X["lang"])))
lang_dict = {name : i for i, name in lang_list}
X.loc[:, "lang_num"] = X["lang"].map(lambda x: lang_dict[x]).astype(int)
X.drop(["name"], axis=1, inplace=True)
X = X[[
    "statuses_count",
    "followers_count",
    "friends_count",
    "favourites_count",
    "lang_num",
    "listed_count",
    "geo_enabled",
    "profile_use_background_image"
                    ]]

X = X.replace(np.nan, 0) 
train_X, test_X, train_y, test_y = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, train_size=0.8, test_size=0.2, random_state=0)
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=8))
model.add(Dense(64, input_dim=32, activation='relu'))
model.add(Dense(64, input_dim=64, activation='relu'))
model.add(Dense(32,input_dim=64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
    
model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
history = model.fit(train_X, train_y,
                    epochs=100,
                    verbose=1,
                    validation_data=(val_X,val_y))
score = model.evaluate(test_X, test_y, verbose=0)
prediction = model.predict(test_X)

app = Flask(__name__)
def predict1(statuses_count,	followers_count,	friends_count,	favourites_count,	lang_num,	listed_count):
    
    
    
    tst = [int(statuses_count),int(followers_count),int(friends_count),int(favourites_count),int(lang_num),int(listed_count),float(0.0),float(0.0)] 
    test_X = pd.DataFrame(tst).T
    prediction = model.predict(test_X)
    prediction = prediction[0][0]
    l=[((prediction>0.5)*1),score[0],score[1]]
    return l

@app.route('/')
def home():
    return render_template('index.html',prediction=[])

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        message1 = request.form['message1']
        message2 = request.form['message2']
        message3 = request.form['message3']
        message4 = request.form['message4']
        message5 = request.form['message5']
        
    
        if message == "" or message1=="" or message2=="" or message3=="" or message4=="" or message5=="":
            return render_template('index.html', prediction=-1)
        message=int(message)
        message1=int(message1)
        message2=int(message2) 
        message3=int(message3)
        message4=int(message4)
        message5=int(message5)
        
        
            
        pred = predict1(message,message1,message2,message3,message4,message5)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)