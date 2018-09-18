#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 21:36:03 2018
NHLseasonML 2.0 - Output player names
Production Version

ML algo carries a playerId, but not the player's name. This algo generates a
list of player names that can be matched to the playerIds.

Important parameters to check against the production ML algo:
    
    Forecast for players who have scored 35 points in a season
    
This algo grabs more data than it needs... it's just a hack of the ML algo and
is NOT optimized at all! The array that it saves is full of numbers, but
it's the first four columns that are of interest: Season, playerId, firstName,
lastName.

@author: Galloway
"""


import sqlite3
import numpy as np
import pandas as pd

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


def extractlagprediction(player, stat4lag, lag ):
    """
    Similar to extractlag, except that it's to create the array to predict from.
    """

    db_name = "NHLseasonML_seasonstats.db"
    conn = sqlite3.connect(db_name)

    with conn:
        cur = conn.cursor()
        
        cur.execute("SELECT DISTINCT s_skater_summary.seasonId, \
                    s_skater_summary.playerId, s_bio_info.playerFirstName, \
                    s_bio_info.playerLastName, s_skater_summary.playerPositionCode, \
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
        df.columns = ('year', 'playerID', 'firstName','lastName',
                      'position', 'points', 'goals', 'ppPoints',
                      'shots', 'timeOnIcePerGame', 'assists', 'games')
        # deal with the categorical data: Pandas has a function that helps...
        # define names of all categories expected:
        df['position'] = df['position'].astype('category',categories=['C', 'D', 'L', 'R'])
        # append columns for each position
        df = pd.concat([df,pd.get_dummies(df['position'], prefix='position')],axis=1)
        # drop original position column
        df.drop(['position'],axis=1, inplace=True)
        # ensure the results are sorted by year, latest at the top:
        df = df.sort_values(['year'],ascending = False)
        # create a dataframe of shifted values - these are lagged w.r.t. the original dataframe
        dfshift = df.shift(lag)
        # name the columns of the shifted df
        dfshift = dfshift.rename(index=str, columns={stat4lag : str(stat4lag + 'lag')})

        columnindex = df.columns.get_loc(stat4lag)

        df = df.join(dfshift.iloc[:,columnindex]).iloc[lag-1:,:] # This line is distinct from the testing version
        
        return np.array(df)

    
    else:
        temp = np.empty((1,16))
        temp.fill(np.nan)
        df = pd.DataFrame(temp)
        df.columns = ('year', 'playerID', 'firstName', 'lastName', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games', 'position_C', 'position_D', 'position_L', 'position_R', str(stat4lag + 'lag'))
        
        return np.array(df)



#%% 
"""
This section is different than the testing version.
Each player's most recent season has been eliminated (don't worry... their
production for that season still remains as a lagged variable).
This production version of the algo must retrieve the data for predictions
from the database.
"""

# Compile training data
#modelarrayfrom = np.transpose(np.dstack((lagged1,lagged2,lagged3)), (0,2,1))


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


#%% FIGURE OUT WHAT TO SAVE - PROBABLY A NUMPY ARRAY OF ALL PREDICTIONS


np.save('20182019_points_playernames.npy',predictarrayfrom)


