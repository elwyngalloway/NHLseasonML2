#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 21:16:08 2018

@author: Galloway
"""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Masking

# there's some weird stuff going on here with matplotlib...
import matplotlib
matplotlib.use('Qt5Agg')   # generate postscript output by default


#%%
db_name = "NHLseasonML_seasonstats.db"

# connect to our database
conn = sqlite3.connect(db_name)

with conn:
    # get the cursor so we can do stuff
    cur = conn.cursor()
    
    # SQLite statement to retreive the data in question (forwards who have
    # scored more than 50 points in a season), ignoring those from 20172018:
    cur.execute("SELECT playerId FROM s_skater_summary WHERE points > 50 \
                AND playerPositionCode IN ('C', 'F', 'L', 'R') \
                AND seasonID NOT IN (20172018)")
    
    # Put selected playerIds in an array (playerId is a unique identifier)
    data = np.array(cur.fetchall())

# data contains multiple entries for some players (those who have scored
# more than 50 points in multiple seasons) - isolate unique values
players = np.unique(data)

# show number of unique players
print(players.shape[0], "players identified")

# Returns 301 players, AND THIS AGREES with seasonML1

#%% Define a function

def extractlag(player, stat4lag, lag ):
    """
    For now, the stat categories extracted will be hard-coded.
    
    I've also hard-coded the database name and the table name from the database.
    
    player = playerId from the database
    
    stat4lag = name of stat to be lagged (string)
    
    lag = integer value for lagging (must be positive)


    """

    db_name = "NHLseasonML_seasonstats.db"
    
    # connect to our database that will hold everything
    conn = sqlite3.connect(db_name)

    with conn:
        # get the cursor so we can do stuff
        cur = conn.cursor()

        # Notice that the stats extracted are hard-coded...
        cur.execute("SELECT seasonId, points, goals, ppPoints, shots, timeOnIcePerGame, assists, gamesplayed, playerId \
                    FROM s_skater_summary \
                    WHERE seasonId NOT IN (20172018) \
                    AND playerId=?", [player])

        data = cur.fetchall()
    
    if len(data) > 0: # only lag if some data is retreived

        # import data into a dataframe
        df = pd.DataFrame(data)

        # name the columns of df
        df.columns = ('year', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games', 'id')
        # ensure the results are sorted by year, latest at the top:
        df = df.sort_values(['year'],ascending = False) # this step was not necessary for seasonML1 - results were already sorted!
        # create a dataframe of shifted values - these are lagged w.r.t. the original dataframe
        dfshift = df.shift(lag)
        # name the columns of the shifted df
        dfshift.columns = ('yearlag', 'pointslag', 'goalslag', 'ppPointslag', 'shotslag', 'timeOnIcePerGamelag', 'assistslag', 'gameslag', 'idlag')

        # find the index of the column desired for lagging
        columnindex = df.columns.get_loc(stat4lag)

        # append the appropriate column of the shifted df to the end of the original df
        df = df.join(dfshift.iloc[:,columnindex]).iloc[lag:,:]

        #return df # may consider changing to return an array
        return np.array(df)
    
    else: # return NaNs of appropriate shape in case no data is retreived from database
        
        # create an empty array
        temp = np.empty((1,6))
        # fill it with NaNs
        temp.fill(np.nan)
        # convert to a Dataframe
        df = pd.DataFrame(temp)
        # name these columns to match typical output
        df.columns = ('year', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games', 'id','pointslag')
        
        #return df
        return np.array(df)
#%% Use function to extract stats for players identified

if 'lagged1' in locals():
    del(lagged1, lagged2, lagged3)
        
for player in players:
    
    # Start with the first lag
    interim1 = extractlag(int(player),'points',1) # create 2D array of a player's performance
    np.array(pd.DataFrame(interim1).dropna(inplace=True)) # ignore "empty" rows
    
    if interim1.shape[0] > 0:
    
        if 'lagged1' in locals(): # if lagged1 already exists, append the player's results to it
            lagged1 = np.append(lagged1, interim1, axis=0)

        else: # else, create lagged1
            lagged1 = interim1[:]

        
        # Now the second lag
        # Ensure lagged2 will have same shape as lagged1 by making each player's
        # contribution have the same shape for each lag.
        interim = np.zeros_like(interim1) - 1 # The ML can ignore -1

        interim2 = extractlag(int(player),'points',2)
        np.array(pd.DataFrame(interim2).dropna(inplace=True))

        interim[:interim2.shape[0],:] = interim2

        if 'lagged2' in locals():
            lagged2 = np.append(lagged2, interim, axis=0)

        else:
            lagged2 = interim[:,:]

        
        # Now the third lag
        interim = np.zeros_like(interim1) - 1

        interim3 = extractlag(int(player), 'points', 3)
        np.array(pd.DataFrame(interim3).dropna(inplace=True))

        interim[:interim3.shape[0],:] = interim3

        if 'lagged3' in locals():
            lagged3 = np.append(lagged3, interim, axis=0)

        else:
            lagged3 = interim[:,:]


# Check that the shapes of the three arrays are identical:
print(lagged1.shape,lagged2.shape,lagged3.shape)

# Convert these arrays into dataframes for convenience later...
lagged1 = pd.DataFrame(lagged1)
lagged1.columns = ('year', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games', 'id','pointslag')

lagged2 = pd.DataFrame(lagged2)
lagged2.columns = ('year', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games', 'id','pointslag')

lagged3 = pd.DataFrame(lagged3)
lagged3.columns = ('year', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games', 'id','pointslag')

# Lagged arrays ARE OF SAME SIZE as seasonML1
        
#%% Perform some preprocessing

# Tell the function that you want to scale each column's values to be between 0 and 1:
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler to the input data:
scaler = scaler.fit(lagged1.values)

#%% Separate training from target data

# predict from the 20152016 season (lag = 1)
lag1predictfrom = lagged1.loc[lagged1['year'] == 20152016]
# model from the remaining seasons
lag1model = lagged1.loc[lagged1['year'] != 20152016]

# predict from the 20142015 season (lag = 2)
lag2predictfrom = lagged2.loc[lagged1['year'] == 20152016] # the rows of interest are in the same position as those in lagged1
# model from the remaining seasons
lag2model = lagged2.loc[lagged1['year'] != 20152016]

lag3predictfrom = lagged3.loc[lagged1['year'] == 20152016]
lag3model = lagged3.loc[lagged1['year'] != 20152016]

# Ensure the shapes of the arrays are identical for each lag
print(lagged1.shape,lag1predictfrom.shape,lag1model.shape)
print(lagged2.shape,lag2predictfrom.shape,lag2model.shape)
print(lagged3.shape,lag3predictfrom.shape,lag3model.shape)

# The array sizes ARE THE SAME as they were for seasonML1!

#%% Scale the data:

lag1predictfrom = scaler.transform(lag1predictfrom)
lag2predictfrom = scaler.transform(lag2predictfrom)
lag3predictfrom = scaler.transform(lag3predictfrom)

lag1model = scaler.transform(lag1model)
lag2model = scaler.transform(lag2model)
lag3model = scaler.transform(lag3model)

#%% Split into train and test sets:

lag1train, lag1test, lag2train, lag2test, lag3train, lag3test = train_test_split(lag1model, lag2model, lag3model, test_size=0.1) # isolating 10% of data

print("Shape of lag1train:", lag1train.shape)

# The shape IS THE SAME as seasonML1!

#%% Combine three lags into a single 3D array:

# First, for the training data
# define the dimensions of the desired array:
dim1 = lag1train.shape[0]
dim2 = 3
dim3 = lag1train.shape[1]

# initialize the training array
train = np.zeros((dim1, dim2, dim3)) - 1 # set the default value to -1. ML algo can know -1 is to be ignored

# populate the training array
train[:lag1train.shape[0],0,:] = lag1train
train[:lag2train.shape[0],1,:] = lag2train
train[:lag3train.shape[0],2,:] = lag3train


# Now, for the test data
dim1 = lag1test.shape[0]
dim2 = 3
dim3 = lag1test.shape[1]

test = np.zeros((dim1, dim2, dim3)) - 1

test[:lag1test.shape[0],0,:] = lag1test
test[:lag2test.shape[0],1,:] = lag2test
test[:lag3test.shape[0],2,:] = lag3test

# Finally, for the prediction dataset
dim1 = lag1predictfrom.shape[0]
dim2 = 3
dim3 = lag1predictfrom.shape[1]

predictfrom = np.zeros((dim1, dim2, dim3)) - 1 

predictfrom[:lag1predictfrom.shape[0],0,:] = lag1predictfrom
predictfrom[:lag2predictfrom.shape[0],1,:] = lag2predictfrom
predictfrom[:lag3predictfrom.shape[0],2,:] = lag3predictfrom

#%% Split into independant and responding variables:

# Split the training data into independant and responding variables:
train_ind, train_resp = train[:,:,:-1], train[:,:,-1]

# Split test data:
test_ind, test_resp = test[:,:,:-1], test[:,:,-1]

# Split prediction data:
predictfrom_ind, predictfrom_resp = predictfrom[:,:,:-1], predictfrom[:,:,-1]

print("Shape of the array used for prediction:", predictfrom_ind.shape)

# The shape IS THE SAME as seasonML1!

#%% Design and train the LSTM model:
# Design LSTM neural network


# Define the network using the Sequential Keras API
model = Sequential()

# Inform algorithm that -1 represents non-values
model.add(Masking(mask_value=-1, input_shape=(train_ind.shape[1], train_ind.shape[2])))

# Define as LSTM with 9 neurons - not optimized - use 9 because I have 9 statistical categories
model.add(LSTM(9))

# I'm not even sure why I need this part, but it doesn't work without it...
model.add(Dense(train_ind.shape[1]))

# Define a loss function and the Adam optimization algorithm
model.compile(loss='mean_squared_error', optimizer='adam')


# train network
history = model.fit(train_ind, train_resp, epochs=40, batch_size=25, validation_data=(test_ind, test_resp),verbose=0, shuffle=False)

#%% Plot... but need to import matplotlib...
# plot history
#plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()

#%% Make a prediction:

predicted_resp = model.predict(predictfrom_ind)

# Show the shape of the prediction array
print("predicted_resp.shape = ",predicted_resp.shape)

# Shape CHECKS OUT!

#%% Invert scaling:

# Make prediced_resp dimension match predictfrom_ind
predicted_resp = np.expand_dims(predicted_resp,axis=2)


# Invert scaling for forecast

# Add the predicted values to the independent variables used for the prediction
inv_predicted = np.concatenate((predictfrom_ind[:,:,:],predicted_resp), axis=2)

# Invert the 3D array one lag at a time
for i in range(inv_predicted.shape[1]):
    inv_predicted[:,i,:] = scaler.inverse_transform(inv_predicted[:,i,:])
results1 = inv_predicted[:,:,:]

# Isolate the predicted values
inv_predicted_resp = inv_predicted[:,:,-1]


# Now, invert scaling for predictfrom
inv_predictfrom = np.empty_like(predictfrom)
for i in range(predictfrom.shape[1]):
    inv_predictfrom[:,i,:] = scaler.inverse_transform(predictfrom[:,i,:])
actual_resp = inv_predictfrom[:,:,-1]

#%% Evaluate performance:

# calculate RMSE
rmseALL = np.sqrt(mean_squared_error(inv_predicted_resp, actual_resp))
print('Overall RMSE: %.3f' % rmseALL)
rmselag1 = np.sqrt(mean_squared_error(inv_predicted_resp[:,0], actual_resp[:,0]))
print('Lag1 RMSE: %.3f' % rmselag1)
rmselag2 = np.sqrt(mean_squared_error(inv_predicted_resp[:,1], actual_resp[:,1]))
print('Lag2 RMSE: %.3f' % rmselag2)
rmselag3 = np.sqrt(mean_squared_error(inv_predicted_resp[:,2], actual_resp[:,2]))
print('Lag3 RMSE: %.3f' % rmselag3)

# These RMSEs check out. GREAT SUCCESS!



#%% What about an alternate calculation of error:


actual_resp_masked = np.ma.masked_where(actual_resp == -1,actual_resp)
inv_predicted_resp_masked = np.ma.masked_where(actual_resp == -1,inv_predicted_resp)

rmseMEANS = np.sqrt(mean_squared_error(np.mean(inv_predicted_resp_masked,axis=1),
                                       np.mean(actual_resp_masked,axis=1)))
print('Means RMSE: %.3f' % rmseMEANS)

















