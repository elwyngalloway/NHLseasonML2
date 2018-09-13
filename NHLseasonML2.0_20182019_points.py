#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:36:03 2018
NHLseasonML 2.0 - Predicting 20182019 Points
Production Version
Predict the points NHL players will accumulate in the 20182019 season.
Key parameters used for this machine learning algorithm:
    3 years of lag
    6 Layers of Neurons
    6 Neurons in each layer
    50 Epochs
    Batch size of 5
    80 Realizations simulated
    Forecast for players who have scored 35 points in a season
It's worth noting that this algo has been stripped down to some degree. Some
functionality used in development has been removed.
@author: Galloway
"""


import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Masking


#%%
db_name = "NHLseasonML_seasonstats.db"

# connect to our database
conn = sqlite3.connect(db_name)

with conn:
    # get the cursor so we can do stuff
    cur = conn.cursor()
    
    # SQLite statement to retreive the data in question (forwards who have
    # scored more than 30 points in a season):


    cur.execute("SELECT playerId FROM s_skater_summary WHERE points > 35 \
                AND playerPositionCode IN ('C', 'D', 'L', 'R') \
                AND seasonID NOT IN (20182019) ") #should be none for 20182019 anyway
    
    
    # Put selected playerIds in an array (playerId is a unique identifier)
    data = np.array(cur.fetchall())

# data contains multiple entries for some players (those who have scored
# more than 50 points in multiple seasons) - isolate unique values
players = np.unique(data)

# show number of unique players
print(players.shape[0], "players identified")


#%% Define a function to retrieve stats from the database


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
        
        # I want to retrieve stats for a season, plus age, draft position,
        # position code (categorical!), name?
        cur.execute("SELECT DISTINCT s_skater_summary.seasonId, \
                    s_skater_summary.playerId, s_bio_info.playerBirthDate, \
                    s_bio_info.playerDraftOverallPickNo, s_skater_summary.playerPositionCode, \
                    s_skater_summary.points, s_skater_summary.goals, s_skater_summary.ppPoints, \
                    s_skater_summary.shots, s_skater_summary.timeOnIcePerGame, \
                    s_skater_summary.assists, s_skater_summary.gamesplayed \
                    FROM s_skater_summary \
                    INNER JOIN s_bio_info \
                        ON s_bio_info.playerId = s_skater_summary.playerId \
                        AND s_bio_info.seasonId = s_skater_summary.seasonId \
                    WHERE s_skater_summary.playerID = ? \
                    AND s_skater_summary.seasonId NOT IN (20182019)", [player])
        
        

        data = cur.fetchall()
    
    if len(data) > 0: # only lag if some data is retreived

        # import data into a dataframe
        df = pd.DataFrame(data)

        # name the columns of df
        df.columns = ('year', 'playerID', 'birthYear','draftPos', 'position', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games')
        # transform birth date to just birth year, then transform to age, then rename column
        df['birthYear'] = pd.to_datetime(df['birthYear']).dt.year
        df['birthYear'] = df['year'] // 10000 - df['birthYear']
        df = df.rename(index=str, columns={'birthYear': 'age'})
        # deal with the categorical data: Pandas has a function that helps...
        # define names of all categories expected:
        df['position'] = df['position'].astype('category',categories=['C', 'D', 'L', 'R'])
        # append columns for each position
        df = pd.concat([df,pd.get_dummies(df['position'], prefix='position')],axis=1)
        # drop original position column
        df.drop(['position'],axis=1, inplace=True)
        # some players were never drafted - leaves blank in draftPos. Define this as 300
        df['draftPos'].replace('', 300, inplace=True)
        # ensure the results are sorted by year, latest at the top:
        df = df.sort_values(['year'],ascending = False)
        # create a dataframe of shifted values - these are lagged w.r.t. the original dataframe
        dfshift = df.shift(lag)
        # name the columns of the shifted df
        dfshift = dfshift.rename(index=str, columns={stat4lag : str(stat4lag + 'lag')})

        # find the index of the column desired for lagging
        columnindex = df.columns.get_loc(stat4lag)

        # append the appropriate column of the shifted df to the end of the original df
        df = df.join(dfshift.iloc[:,columnindex]).iloc[lag:,:]
        
        #return df # may consider changing to return an array
        return np.array(df)
        #return df
    
    else: # return NaNs of appropriate shape in case no data is retreived from database
        
        # create an empty array
        temp = np.empty((1,16))
        # fill it with NaNs
        temp.fill(np.nan)
        # convert to a Dataframe
        df = pd.DataFrame(temp)
        # name these columns to match typical output
        df.columns = ('year', 'playerID', 'age','draftPos', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games', 'position_C', 'position_D', 'position_L', 'position_R', str(stat4lag + 'lag'))
        
        #return df
        return np.array(df)
        #return df
        
#%% Define a function to retrieve stats from the database


def extractlagprediction(player, stat4lag, lag ):
    """
    Similar to extractlag, except that it's to create the array to predict from.
    """

    db_name = "NHLseasonML_seasonstats.db"
    conn = sqlite3.connect(db_name)

    with conn:
        cur = conn.cursor()
        
        cur.execute("SELECT DISTINCT s_skater_summary.seasonId, \
                    s_skater_summary.playerId, s_bio_info.playerBirthDate, \
                    s_bio_info.playerDraftOverallPickNo, s_skater_summary.playerPositionCode, \
                    s_skater_summary.points, s_skater_summary.goals, s_skater_summary.ppPoints, \
                    s_skater_summary.shots, s_skater_summary.timeOnIcePerGame, \
                    s_skater_summary.assists, s_skater_summary.gamesplayed \
                    FROM s_skater_summary \
                    INNER JOIN s_bio_info \
                        ON s_bio_info.playerId = s_skater_summary.playerId \
                        AND s_bio_info.seasonId = s_skater_summary.seasonId \
                    WHERE s_skater_summary.playerID = ? \
                    AND s_skater_summary.seasonId NOT IN (20182019)", [player])

        data = cur.fetchall()
    
    if len(data) > 0:

        df = pd.DataFrame(data)

        df.columns = ('year', 'playerID', 'birthYear','draftPos', 'position', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games')
        df['birthYear'] = pd.to_datetime(df['birthYear']).dt.year
        df['birthYear'] = df['year'] // 10000 - df['birthYear']
        df = df.rename(index=str, columns={'birthYear': 'age'})
        df['position'] = df['position'].astype('category',categories=['C', 'D', 'L', 'R'])
        df = pd.concat([df,pd.get_dummies(df['position'], prefix='position')],axis=1)
        df.drop(['position'],axis=1, inplace=True)
        df['draftPos'].replace('', 300, inplace=True)
        df = df.sort_values(['year'],ascending = False)
        dfshift = df.shift(lag)
        dfshift = dfshift.rename(index=str, columns={stat4lag : str(stat4lag + 'lag')})

        columnindex = df.columns.get_loc(stat4lag)

        df = df.join(dfshift.iloc[:,columnindex]).iloc[lag-1:,:] # This line is distinct from the testing version
        
        return np.array(df)

    
    else:
        temp = np.empty((1,16))
        temp.fill(np.nan)
        df = pd.DataFrame(temp)
        df.columns = ('year', 'playerID', 'age','draftPos', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games', 'position_C', 'position_D', 'position_L', 'position_R', str(stat4lag + 'lag'))
        
        return np.array(df)

#%% Use function to extract stats for players identified
        
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



#%% 
"""
This section is different than the testing version.
Each player's most recent season has been eliminated (don't worry... their
production for that season still remains as a lagged variable).
This production version of the algo must retrieve the data for predictions
from the database.
"""

# Compile training data
modelarrayfrom = np.transpose(np.dstack((lagged1,lagged2,lagged3)), (0,2,1))


#Compile the predict-from data
# Go back to the database, retrieve the data for the predict-from array
for player in players:
    
    # Start with the first lag
    interim1 = extractlagprediction(int(player),'points',1) # create 2D array of a player's performance
    np.array(pd.DataFrame(interim1).dropna(inplace=True)) # ignore "empty" rows
    
    if interim1.shape[0] > 0:
    
        if 'lagged1pred' in locals(): # if lagged1 already exists, append the player's results to it
            lagged1pred = np.append(lagged1pred, interim1, axis=0)

        else: # else, create lagged1
            lagged1pred = interim1[:]

        
        # Now the second lag
        # Ensure lagged2 will have same shape as lagged1 by making each player's
        # contribution have the same shape for each lag.
        interim = np.zeros_like(interim1) - 999 # Identify missing data as -999

        interim2 = extractlagprediction(int(player),'points',2)
        np.array(pd.DataFrame(interim2).dropna(inplace=True))

        interim[:interim2.shape[0],:] = interim2

        if 'lagged2pred' in locals():
            lagged2pred = np.append(lagged2pred, interim, axis=0)

        else:
            lagged2pred = interim[:,:]

        
        # Now the third lag
        interim = np.zeros_like(interim1) - 999

        interim3 = extractlagprediction(int(player), 'points', 3)
        np.array(pd.DataFrame(interim3).dropna(inplace=True))

        interim[:interim3.shape[0],:] = interim3

        if 'lagged3pred' in locals():
            lagged3pred = np.append(lagged3pred, interim, axis=0)

        else:
            lagged3pred = interim[:,:]

lag1predictfrom = lagged1pred[lagged1pred[:,0] == 20172018]
lag2predictfrom = lagged2pred[lagged1pred[:,0] == 20172018]
lag3predictfrom = lagged3pred[lagged1pred[:,0] == 20172018]

predictarrayfrom = np.transpose(np.dstack((lag1predictfrom, \
                                           lag2predictfrom, \
                                           lag3predictfrom)), (0,2,1))

predictarrayfrom[np.isnan(predictarrayfrom)]=-999

#%% Let's harness things from here on. Define a function that separates the
#   data into training and testing sets; trains the model; predicts; evaluates
#   prediction quality


def modelrun(modelfrom, predictfrom):
    
    """
    
    """
    
    # We need to address the missing values (-999s) before scaling.
    # Create masks of the modelfrom and predictfrom
    modelfrommask = np.ma.masked_equal(modelfrom,-999).mask
    predictfrommask = np.ma.masked_equal(predictfrom,-999).mask
    # Use them to reassign -999s as max stat value - this keeps -999 from affection the scaling
    modelfrom[modelfrommask] = (np.ones_like(modelfrom)*np.max(modelfrom,(0,1)))[modelfrommask]
    predictfrom[predictfrommask] = (np.ones_like(predictfrom)*np.max(modelfrom,(0,1)))[predictfrommask] # Slightly different than testing version
    
    
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
    
    #Design and train the LSTM model:
    # Design LSTM neural network
    
    # Define the network using the Sequential Keras API
    model = Sequential()
    
    # Inform algorithm that 0 represents non-values (values of -1 were scaled to 0!)
    model.add(Masking(mask_value=-999, input_shape=(train_ind.shape[1], train_ind.shape[2])))
    
    # Define as LSTM
    model.add(LSTM(6, return_sequences=True))
    model.add(LSTM(6, return_sequences=True))
    model.add(LSTM(6, return_sequences=True))
    model.add(LSTM(6, return_sequences=True))
    model.add(LSTM(6, return_sequences=True))
    model.add(LSTM(6))
    
    # I'm not even sure why I need this part, but it doesn't work without it...
    model.add(Dense(train_ind.shape[1]))
    
    # Define a loss function and the Adam optimization algorithm
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    
    # train network
    history = model.fit(train_ind, train_resp, epochs=50, batch_size=5, validation_data=(test_ind, test_resp),verbose=0, shuffle=False)


    # Make a prediction:    
    predicted_resp = model.predict(predictfrom_ind)
    
    # Invert scaling:
    
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
    inv_predicted_resp = test_predicted[:,:,-1]
    
    # Return results (predicted responding variables):
    return inv_predicted_resp


#%% Run iterations:

numiters = 80
for i in range(numiters):
    print("Working on prediction " + str(i+1) + "/" + str(numiters) + " = " + str(int(i/numiters*100)) + "% complete")
    if i == 0:
        result = np.expand_dims(modelrun(modelarrayfrom, predictarrayfrom), axis=2)
    else:
        result = np.concatenate((result,np.expand_dims(modelrun(modelarrayfrom, predictarrayfrom), axis=2)),axis=2)

#%% FIGURE OUT WHAT TO SAVE - PROBABLY A NUMPY ARRAY OF ALL PREDICTIONS


np.save('20182019_points_L06N06E50B05.npy',result)


