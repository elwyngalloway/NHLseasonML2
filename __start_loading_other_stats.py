#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 20:36:06 2018

@author: Galloway


I'm starting to mess around with how to import additional stats, including
year of birth, draft position, player position

Also, this brings playerID back into the flow. Prior tests showed that it
didn't really have an impact. How will the predictions be reconnected to
players if ID is not carried through?



"""


import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Masking

import matplotlib.pyplot as plt
from scipy import stats


#%%
db_name = "NHLseasonML_seasonstats.db"

# connect to our database
conn = sqlite3.connect(db_name)

with conn:
    # get the cursor so we can do stuff
    cur = conn.cursor()
    
    # SQLite statement to retreive the data in question (forwards who have
    # scored more than 50 points in a season):


    cur.execute("SELECT playerId FROM s_skater_summary WHERE points > 50 \
                AND playerPositionCode IN ('C', 'F', 'L', 'R') \
                AND seasonID NOT IN (20172018) ")
    
    
    # Put selected playerIds in an array (playerId is a unique identifier)
    data = np.array(cur.fetchall())

# data contains multiple entries for some players (those who have scored
# more than 50 points in multiple seasons) - isolate unique values
players = np.unique(data)

# show number of unique players
print(players.shape[0], "players identified")


#%% Define a function - modifying from ML1... still in progress


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
                    AND s_skater_summary.seasonId NOT IN (20172018)", [player])
        
        

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
        df = df.sort_values(['year'],ascending = False) # this step was not necessary for seasonML1 - results were already sorted!
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
lagged1 = lagged1.rename(index=str, columns={0: 'year'})

lagged2 = pd.DataFrame(lagged2)
lagged2 = lagged2.rename(index=str, columns={0: 'year'})

lagged3 = pd.DataFrame(lagged3)
lagged3 = lagged3.rename(index=str, columns={0: 'year'})

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

## predict from the 20152016 season (lag = 1)
#lag1predictfrom = lagged1.loc[lagged1['year'] == 20162017]
## model from the remaining seasons
#lag1model = lagged1.loc[lagged1['year'] != 20162017]
#
## predict from the 20142015 season (lag = 2)
#lag2predictfrom = lagged2.loc[lagged1['year'] == 20162017] # the rows of interest are in the same position as those in lagged1
## model from the remaining seasons
#lag2model = lagged2.loc[lagged1['year'] != 20162017]
#
#lag3predictfrom = lagged3.loc[lagged1['year'] == 20162017]
#lag3model = lagged3.loc[lagged1['year'] != 20162017]



# This array contains all data needed test and train the model
modelarrayfrom = np.transpose(np.dstack((np.array(lag1model),
                                    np.array(lag2model),
                                    np.array(lag3model))), (0,2,1))

# This array is the one that will be predicted from:
predictarrayfrom = np.transpose(np.dstack((np.array(lag1predictfrom),
                                      np.array(lag2predictfrom),
                                      np.array(lag3predictfrom))), (0,2,1))



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
    
    #Design and train the LSTM model:
    # Design LSTM neural network
    
    # Define the network using the Sequential Keras API
    model = Sequential()
    
    # Inform algorithm that 0 represents non-values (values of -1 were scaled to 0!)
    model.add(Masking(mask_value=-999, input_shape=(train_ind.shape[1], train_ind.shape[2])))
    
    # Define as LSTM with 8 neurons - not optimized - use 8 because I have 8 statistical categories
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

    # plot history
#    plt.plot(history.history['loss'], label='train')
#    plt.plot(history.history['val_loss'], label='test')
#    plt.xlabel('Epoch')
#    plt.ylabel('Loss')
#    plt.legend()
#    plt.show()

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
#del(result)
numiters = 10
for i in range(numiters):
    print("Working on prediction " + str(i+1) + "/" + str(numiters) + " = " + str(int(i/numiters*100)) + "% complete")
    if i == 0:
        result = np.expand_dims(modelrun(modelarrayfrom, predictarrayfrom), axis=2)
    else:
        result = np.concatenate((result,np.expand_dims(modelrun(modelarrayfrom, predictarrayfrom), axis=2)),axis=2)
        
#%%   

# Evaluate performance:

# Retrieve the responding variables for predictarrayfrom
actual = predictarrayfrom[:,0,-1]

# Find the mask
resultmask = np.ma.masked_less(result,2).mask

# result.shape = [player, lag, iteration]


# Create an alternate measure of error: use mean of the lags for each player
# as the prediction. Calculate the RMSE of these means.         
RMSEmeans =np.empty((result.shape[2]))

meanresult = np.zeros((result.shape[0],result.shape[2]))

for iteration in range(result.shape[2]):
    for player in range(result.shape[0]):
        meanresult[player,iteration] = np.mean(result[player,:,iteration][np.ma.masked_greater(result[player,:,iteration],2).mask])
    
    RMSEmeans[iteration] = np.sqrt(mean_squared_error(meanresult[:,iteration],actual))


# For now, I think the best representation of the error is the RMSE for
# the mean of the the lag estimates. Report this as error.
error = np.mean(RMSEmeans)

print("Overall error: " + str(error))

#np.save('./results/LAG3_POINTS50/plus_stats/LSTM15-MSE_ADAM-epo128_batch25.npy',result)

# something's up with the error... rookie points are still being forecast for years not played...
# actually, that's expected! They are still forecast, but we want to ignore them during
# error calculations. I adjusted the threshold for the mask to 2, and that should make a difference.

# Plot some results like this:

fig1 = plt.figure(figsize=(5,5))
az = fig1.add_subplot(1,1,1)
az.scatter(actual,np.mean(meanresult, axis=1),c="b", s=10)

az.plot([0,50,120],[0,50,120])
plt.ylim(-5,110)
plt.xlim(-5,110)
plt.xlabel('Actual Results')
plt.ylabel('Predicted Results')
plt.title('20162017 Alt Stats:\nL05N06E50B05', fontsize=16)
plt.grid(True)
plt.text(5,95,str('RMSE = '+str(round(float(error),2))),fontsize=16)

#%% Try using percentile as an error? Should give an indication of the relative
# ranking, which is what we're really after...

# for a set of results, transform the predicted score into a percentile
# Retrieve the responding variables for predictarrayfrom
actual = predictarrayfrom[:,0,-1]

# Find the mask
resultmask = np.ma.masked_less(result,1).mask

# Create an alternate measure of error: use mean of the lags for each player      

meanresult = np.zeros((result.shape[0],result.shape[2]))

for iteration in range(result.shape[2]):
    for player in range(result.shape[0]):
        meanresult[player,iteration] = np.mean(result[player,:,iteration][np.ma.masked_greater(result[player,:,iteration],2).mask])
    
meanresultpercentile = np.zeros_like(meanresult)

for iteration in range(result.shape[2]):
    for player in range(result.shape[0]):
        meanresultpercentile[player,iteration] = stats.percentileofscore(meanresult[:,iteration], meanresult[player,iteration])

# for the actual results, transform them into percentiles
actualpercentile = np.zeros_like(actual)

for player in range(actual.shape[0]):
    actualpercentile[player] = stats.percentileofscore(actual,actual[player])

# calculate the RMSE of the percentiles
RMSEpercentiles = np.empty((result.shape[2]))


for iteration in range(result.shape[2]):
    RMSEpercentiles[iteration] = np.sqrt(mean_squared_error(meanresultpercentile[:,iteration],actualpercentile))


errorpercentile = np.mean(RMSEpercentiles)

# plot the predicted and actual percentiles
# plot one realizaiton


fig = plt.figure(figsize=(5,5))

az = fig.add_subplot(1,1,1)
az.scatter(actualpercentile,np.mean(meanresultpercentile, axis=1),c="b", s=10)
#az.scatter(actualpercentile,meanresultpercentile[:,0],c="b", s=10)
az.plot([0,50,120],[0,50,120])
plt.ylim(-5,110)
plt.xlim(-5,110)
plt.xlabel('Actual Results')
plt.ylabel('Predicted Results')
plt.title('20162017 Alt Stats:\nPercentile L05N06E50B05', fontsize=16)
plt.grid(True)
plt.text(5,95,str('RMSE = '+str(round(float(errorpercentile),2))),fontsize=16)