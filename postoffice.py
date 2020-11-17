#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import datetime as dt
import numpy as np
import matplotlib
import os

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from tensorflow.python import keras


# In[3]:


def file_len(filename):
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i


# In[4]:


def file_read(path, numberOfRows):
    indexCounter = 0
    with open(path,'r') as file:
        nRows = numberOfRows
        nColumns = 4
        dataset = np.zeros(shape=(nRows, nColumns))
        times_arrival = []
        for line in file:
            try:
                dataInstance = line.split(',')
                time_arrival = dataInstance[1] #splits the line at the comma and takes the first bit
                time_arrival = dt.datetime.strptime(time_arrival, '%H:%M')
                hour_arrival = time_arrival.hour
                minute_arrival = time_arrival.minute
                waitingMinutes = dataInstance[2]
                serviceMinutes = dataInstance[3]

                times_arrival.append(time_arrival)
                dataset[indexCounter] = [hour_arrival, minute_arrival, waitingMinutes, serviceMinutes]
                indexCounter = indexCounter + 1
            except:
                #print('index' + str(indexCounter) + 'error')
                pass
    return dataset, times_arrival


# In[6]:


filenames = []
rootFilePath = './PostOfficeDataCsv/'
completeDataset = pd.DataFrame()

for post_office in range(3):
    for week in range(4):
        if(post_office == 0 and week == 0):
            i = 2
        else:
            i = 0
            
        for day in range(i,5):
            filename = 'PostOffice' + str(post_office + 1) + 'Week' + str(week + 1) + 'Day' + str(day + 1)
            fullPath = rootFilePath + filename + '.csv'
            filenames.append(fullPath)
            
            numberOfRows = file_len(fullPath) - 1
            print ('Reading ' + filename + 'that contains' + str(numberOfRows) + ' entries')
            tempFeatures, tempArrivalTimes = file_read(rootFilePath + filename + '.csv', numberOfRows)
            dfTempFeatures = pd.DataFrame(np.array(tempFeatures), columns=['hour', 'minutes', 'waitingTime', 'serviceTime'])
            dfTempArrivalTimes = pd.DataFrame(np.array(tempArrivalTimes), columns=['arrivalTime'])
            
            timeLeavingTheQueue = []
            for arrivalTime in range(numberOfRows):
                timeLeavingTheQueue.append(dfTempArrivalTimes.at[arrivalTime, 'arrivalTime'] + pd.Timedelta(minutes = dfTempFeatures.at[arrivalTime, 'waitingTime']))
            dftimeLeavingTheQueue = pd.DataFrame(np.array(timeLeavingTheQueue), columns=['timeLeavingTheQueue'])

            waitingPeople = np.zeros(numberOfRows)
            for i in range(numberOfRows):
                for j in range(i):
                    if (dfTempArrivalTimes.at[i, 'arrivalTime'] < dftimeLeavingTheQueue.at[j, 'timeLeavingTheQueue']):
                        waitingPeople[i] += 1
            dfWaitingPeople = pd.DataFrame(np.array(waitingPeople), columns=['waitingPeople'])
            
            dayOfWeek = np.zeros(numberOfRows)
            for i in range(numberOfRows):
                dayOfWeek[i] = day
            dfDayOfWeek = pd.DataFrame(np.array(dayOfWeek), columns=['dayOfWeek'])
            
            dfWaitingPeople['waitingPeople'] = dfWaitingPeople['waitingPeople'].astype(int)
            dfTempFeatures['hour'] = dfTempFeatures['hour'].astype(int)
            dfTempFeatures['minutes'] = dfTempFeatures['minutes'].astype(int)
            dfDayOfWeek['dayOfWeek'] = dfDayOfWeek['dayOfWeek'].astype(int)
    
            tempDataset = pd.concat([dfTempFeatures, dfWaitingPeople, dfDayOfWeek], axis=1)
        
            completeDataset = pd.concat([completeDataset, tempDataset], axis=0)
          
completeDataset = completeDataset.reset_index(drop = True)
print(completeDataset.shape[0])


# In[8]:


print(f'The dataset has {completeDataset.shape[0]} rows and {completeDataset.shape[1]} columns.')


# In[9]:


completeDataset


# In[10]:


mu = completeDataset["waitingPeople"].mean()  # mean of distribution
sigma = completeDataset["waitingPeople"].std()  # standard deviation of distribution
x = completeDataset["waitingPeople"]

num_bins = 111

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=1)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.plot(bins, y, '--')
ax.set_xlabel('Number of people waiting when joining the queue')
ax.set_ylabel('Probability density')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
# plt.savefig('./plots/waitingPeopleHistogram.pdf')


# In[11]:


# data to be plotted
mu = completeDataset["waitingTime"].mean()  # mean of distribution
sigma = completeDataset["waitingTime"].std()  # standard deviation of distribution
x = completeDataset["waitingTime"]

num_bins = 33

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=1)

# add a 'best fit' line
y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
     np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
ax.plot(bins, y, '--')
ax.set_xlabel('Customer waiting time (mins)')
ax.set_ylabel('Probability density')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()


# In[12]:


workingCopyDataset = completeDataset
workingCopyDataset.drop(['serviceTime'], axis=1);


# In[13]:


# mean encoding for regression output
def mean_encoder_regression(input_vector, output_vector):
    assert len(input_vector) == len(output_vector)
    numberOfRows = len(input_vector)

    temp = pd.concat([input_vector, output_vector], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=input_vector.name)[output_vector.name].agg(["mean", "count"])
    
    print(averages)
    return_vector = pd.DataFrame(0, index=np.arange(numberOfRows), columns={'feature'})

    
    for i in range(numberOfRows):
        return_vector.iloc[i] = averages['mean'][input_vector.iloc[i]]
        
    return return_vector


# In[14]:


encoded_input_vector_hour = mean_encoder_regression(workingCopyDataset['hour'], workingCopyDataset['waitingTime'])
encoded_input_vector_hour.columns = ['hour']
encoded_input_vector_minutes = mean_encoder_regression(workingCopyDataset['minutes'], workingCopyDataset['waitingTime'])
encoded_input_vector_minutes.columns = ['minutes']
encoded_input_vector_dayOfWeek = mean_encoder_regression(workingCopyDataset['dayOfWeek'], workingCopyDataset['waitingTime'])
encoded_input_vector_dayOfWeek.columns = ['dayOfWeek']


# In[15]:


X = pd.concat([encoded_input_vector_hour['hour'], encoded_input_vector_minutes['minutes'], pd.DataFrame(workingCopyDataset['waitingPeople']), encoded_input_vector_dayOfWeek['dayOfWeek']], axis=1)
y = workingCopyDataset['waitingTime']


# In[16]:


X


# In[17]:


trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.2, random_state=42)
print(trainX.shape, trainy.shape)
print(testX.shape, testy.shape)


# In[18]:


def scale_input(X, means, stds):
    return (X - means) / stds


# In[19]:


meansX = trainX.mean(axis=0)
stdsX = trainX.std(axis=0) + 1e-10


# In[20]:


trainX_scaled = scale_input(trainX, meansX, stdsX)
testX_scaled = scale_input(testX, meansX, stdsX)


# In[21]:


trainX_scaled


# In[28]:


inputVariables = 4
nn_model = keras.models.Sequential()
nn_model.add(keras.layers.Dense(12, input_dim=inputVariables, kernel_initializer='normal', activation='relu'))
nn_model.add(keras.layers.Dense(8, activation='relu'))
nn_model.add(keras.layers.Dense(4, activation='relu'))
nn_model.add(keras.layers.Dense(1, activation='linear'))
nn_model.summary()

nn_model.compile(loss='mae', optimizer='adam')


# In[29]:


numberOfEpochs = 500
batchSize = 256
history = nn_model.fit(trainX_scaled, trainy, epochs=numberOfEpochs, batch_size=batchSize, verbose=1, validation_split=0.2)


# In[30]:


neural_network_predict_test = nn_model.predict(testX_scaled)
neural_network_predict_test


# In[31]:


neural_network_predict_test = nn_model.predict(testX_scaled)
neural_net_mae = mean_absolute_error(neural_network_predict_test,testy)
neural_net_mae


# In[32]:


from sklearn.ensemble import RandomForestRegressor

randomForestRegressorModel = RandomForestRegressor(n_estimators=100, random_state=0)
randomForestRegressorModel.fit(trainX_scaled,trainy)


# In[33]:


random_forest_test_predict = randomForestRegressorModel.predict(testX_scaled)
random_forest_mae = mean_absolute_error(random_forest_test_predict,testy)
random_forest_mae


# In[26]:


from platform import python_version

print(python_version())


# In[ ]:




