#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LSTM


# In[30]:


data = pd.read_csv('expanded_data1.csv')


# In[31]:


data.head()


# In[32]:


data.describe()


# In[33]:


# len(data[data['status'] == 0])  Alternative Method

data['status'].value_counts()


# In[34]:


X =  data.drop(['name', 'status'] , axis =1)
Y = data['status']


# In[35]:


X_train , X_test , Y_train , Y_test =  train_test_split(X,Y, test_size=0.2 , random_state=42)


# In[36]:


scaler = StandardScaler()


# In[37]:


scaler.fit(X_train)


# In[38]:


X_train = scaler.transform(X_train)

X_test  = scaler.transform(X_test)


# # Support Vector Machine (SVM)

# In[39]:


model =  svm.SVC(kernel='rbf')


# In[40]:


model.fit(X_train,Y_train)


# In[41]:


X_train_prediction = model.predict(X_train)


# In[42]:


train_accuracy_score = accuracy_score(Y_train, X_train_prediction)


# In[43]:


X_test_prediction = model.predict(X_test)


# In[44]:


test_accuracy_score =  accuracy_score(Y_test , X_test_prediction)


# In[45]:


print("Train Accuracy: " ,train_accuracy_score)
print("Test Accuracy: " ,test_accuracy_score)


# In[46]:


model1 = Sequential([
Dense(64, activation='relu' ,input_shape= (X_train.shape[1],)),
Dense(64, activation = 'relu'),
Dense(1, activation = 'sigmoid')
])


# In[47]:


model1.compile(optimizer = 'adam' , loss='binary_crossentropy' , metrics=['accuracy'])


# In[48]:


model1.fit(X_train, Y_train , epochs= 20 , batch_size = 15 , verbose =1)


# In[49]:


test_loss , test_accuracy =  model1.evaluate(X_test, Y_test)
print(test_loss , test_accuracy)


# In[50]:


model_cnn = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),
    Conv1D(128, 3, activation='relu'),
    MaxPooling1D(2),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])


# In[51]:


model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[52]:


model_cnn.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), Y_train, epochs=20, batch_size=15, verbose=1)


# In[53]:


test_loss_cnn, test_accuracy_cnn = model_cnn.evaluate(X_test.reshape((X_test.shape[0], X_test.shape[1], 1)), Y_test)
print('CNN Model Test Accuracy:', test_accuracy_cnn)


# In[54]:


randomForest = RandomForestClassifier(n_estimators=100,  random_state=42)


# In[55]:


randomForest.fit(X_train , Y_train)


# In[56]:


y_pred_rf = randomForest.predict(X_test)


# In[57]:


accuracy_rf =  accuracy_score(y_pred_rf , Y_test)
print('Random Forest Classifier Test Accuracy:', accuracy_rf)


# In[ ]:




