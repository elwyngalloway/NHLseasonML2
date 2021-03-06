#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The first half hour of my ML course informed me that I was making a couple
obvious mistakes when preparing seasonML1.

1. I did not treat categorical data correctly (not a problem in the
   production implementation of the algorithm, but it would have been
   if I had included any categorical data!)
2. Player ID should not be included as a data point in the model or for
   prediction. Ideally, it would come through the flow with the data somehow...

Does including Player ID affect the prediction? Negatively or Positively?
Let's find out!

... multiple runs of each showed negligible difference. Meh.

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

# there's some weird stuff going on here with matplotlib..
#matplotlib.use('Qt5Agg')   # generate postscript output by default
import matplotlib


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
        cur.execute("SELECT seasonId, points, goals, ppPoints, shots, timeOnIcePerGame, assists, gamesplayed \
                    FROM s_skater_summary \
                    WHERE seasonId NOT IN (20172018) \
                    AND playerId=?", [player])

        data = cur.fetchall()
    
    if len(data) > 0: # only lag if some data is retreived

        # import data into a dataframe
        df = pd.DataFrame(data)

        # name the columns of df
        df.columns = ('year', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games')
        # ensure the results are sorted by year, latest at the top:
        df = df.sort_values(['year'],ascending = False) # this step was not necessary for seasonML1 - results were already sorted!
        # create a dataframe of shifted values - these are lagged w.r.t. the original dataframe
        dfshift = df.shift(lag)
        # name the columns of the shifted df
        dfshift.columns = ('yearlag', 'pointslag', 'goalslag', 'ppPointslag', 'shotslag', 'timeOnIcePerGamelag', 'assistslag', 'gameslag')

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
        df.columns = ('year', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games','pointslag')
        
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
        interim = np.zeros_like(interim1) - 999 # Identify missing data as -999

        interim2 = extractlag(int(player),'points',2)
        np.array(pd.DataFrame(interim2).dropna(inplace=True))

        interim[:interim2.shape[0],:] = interim2

        if 'lagged2' in locals():
            lagged2 = np.append(lagged2, interim, axis=0)

        else:
            lagged2 = interim[:,:]

        
        # Now the third lag
        interim = np.zeros_like(interim1) - 999

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
lagged1.columns = ('year', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games', 'pointslag')

lagged2 = pd.DataFrame(lagged2)
lagged2.columns = ('year', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games', 'pointslag')

lagged3 = pd.DataFrame(lagged3)
lagged3.columns = ('year', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games', 'pointslag')


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


#%% updated flow here... let's amalgamate the lags into a single array

# This array contains all data needed test and train the model
modelfrom = np.transpose(np.dstack((np.array(lag1model), np.array(lag2model), np.array(lag3model))), (0,2,1))

# This array is the one that will be predicted from:
predictfrom = np.transpose(np.dstack((np.array(lag1predictfrom), np.array(lag2predictfrom), np.array(lag3predictfrom))), (0,2,1))


# We need to address the missing values (-999s) before scaling.
# Create masks of the modelfrom and predictfrom
modelfrommask = np.ma.masked_equal(modelfrom,-999).mask
predictfrommask = np.ma.masked_equal(predictfrom,-999).mask
# Use them to reassign -999s as max stat value
modelfrom[modelfrommask] = (np.ones_like(modelfrom)*np.max(modelfrom,(0,1)))[modelfrommask]
predictfrom[predictfrommask] = (np.ones_like(predictfrom)*np.max(predictfrom,(0,1)))[predictfrommask]


#  Apply the 3D scaler:

scaler = MinMaxScaler(feature_range=(0, 1))

# Design the scaler:
# (this flattens the 3D array into 2D, applies determines scaler, then re-stacks in 3D)

scaler = scaler.fit(modelfrom.reshape(-1,modelfrom.shape[2]))

# Apply the scaler:
modelfrom_scaled = scaler.transform(modelfrom.reshape(-1, modelfrom.shape[2])).reshape(modelfrom.shape)
predictfrom_scaled = scaler.transform(predictfrom.reshape(-1, predictfrom.shape[2])).reshape(predictfrom.shape)

# Return the missing values to -999
modelfrom[modelfrommask] = -999
predictfrom[predictfrommask] = -999



# Split into test and training sets:
train, test = train_test_split(modelfrom_scaled,test_size=0.1)


# Split into independant and responding variables:

# Split the training data into independant and responding variables:
train_ind, train_resp = train[:,:,:-1], train[:,:,-1]

# Split test data:
test_ind, test_resp = test[:,:,:-1], test[:,:,-1]

# Split prediction data:
predictfrom_ind, predictfrom_resp = predictfrom_scaled[:,:,:-1], predictfrom_scaled[:,:,-1]

print("Shape of the array used for prediction:", predictfrom_ind.shape)

# The shape IS THE SAME as seasonML1!

#%% Design and train the LSTM model:
# Design LSTM neural network


# Define the network using the Sequential Keras API
model = Sequential()

# Inform algorithm that 0 represents non-values (values of -1 were scaled to 0!)
model.add(Masking(mask_value=-999, input_shape=(train_ind.shape[1], train_ind.shape[2])))

# Define as LSTM with 8 neurons - not optimized - use 8 because I have 8 statistical categories
model.add(LSTM(8))

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

inv_predicted = scaler.inverse_transform(inv_predicted.reshape(-1, inv_predicted.shape[2])).reshape(inv_predicted.shape)

# Make sure the missing data is ignored
test_predicted = np.empty_like(inv_predicted)
test_predicted[~predictfrommask] = inv_predicted[~predictfrommask]

# Isolate the predicted values
inv_predicted_resp = inv_predicted[:,:,-1]


# Now, invert scaling for predictfrom
inv_predictfrom = scaler.inverse_transform(predictfrom_scaled.reshape(-1, predictfrom_scaled.shape[2])).reshape(predictfrom_scaled.shape)

# Make sure the missing data is ignored
test_predictfrom = np.empty_like(inv_predictfrom)
test_predictfrom[~predictfrommask] = inv_predictfrom[~predictfrommask]

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

















