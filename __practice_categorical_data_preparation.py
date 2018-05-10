#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:16:54 2018

@author: Galloway
"""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

db_name = "NHLseasonML_seasonstats.db"

# connect to our database that will hold everything, or create it if it doesn't exist
conn = sqlite3.connect(db_name)

#%%
# retrieve some data from the database, then place a bit of it in a dataframe

with conn:
        # get the cursor so we can do stuff
        cur = conn.cursor()

        # extract a number of columns from s_skater_summary
        cur.execute("SELECT seasonId, playerPositionCode, assists, goals, gamesplayed, playerId FROM s_skater_summary")

        data = cur.fetchall()
        
orig = np.array(pd.DataFrame(data[:50]))
X = np.array(pd.DataFrame(data))
#%%
# transform categorical strings into numerical values -> playerPositionCode
labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])

# now create dummy variables for each categorical value
onehotencoder = OneHotEncoder(categorical_features = [1])
X2 = onehotencoder.fit_transform(X).toarray()

#%%
# check to see that number of additional columns match expectations

print('number of categories = ' + str(len(np.unique(orig[:,1]))))
print('number of addiional columns = ' + str((X2.shape[1] - orig.shape[1])))

if len(np.unique(orig[:,1])) != (X2.shape[1] - orig.shape[1] + 1):
    print('uh-oh.....')
else:
    print('everything is as expected!')
    
    
#%%














