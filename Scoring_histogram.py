#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 19:33:26 2018

@author: Galloway
"""

# Let's see how scoring changed from 20142015 through 20172018



import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from scipy import stats

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

#%% Query the database

db_name = "NHLseasonML_seasonstats.db"

# connect to our database
conn = sqlite3.connect(db_name)


with conn:
    # get the cursor so we can do stuff
    cur = conn.cursor()
    
    # SQLite statement to retreive the data in question (forwards who have
    # scored more than 50 points in a season):
    cur.execute("SELECT points, seasonId\
                    FROM s_skater_summary \
                    WHERE seasonId IN (20142015, 20152016, 20162017, 20172018)")
    
    # Put selected playerIds in an array (playerId is a unique identifier)
    data = np.array(cur.fetchall())

#%%
    
p20142015 = np.sort(data[np.where(data[:,1] == 20142015)][:,0])
p20152016 = np.sort(data[np.where(data[:,1] == 20152016)][:,0])
p20162017 = np.sort(data[np.where(data[:,1] == 20162017)][:,0])
p20172018 = np.sort(data[np.where(data[:,1] == 20172018)][:,0])

#%% plot histograms

sns.distplot(p20142015[-100:], label='20142015', axlabel='Points', kde=False)
sns.distplot(p20152016[-100:], label='20152016', axlabel='Points', kde=False)
sns.distplot(p20162017[-100:], label='20162017', axlabel='Points', kde=False)
sns.distplot(p20172018[-100:], label='20172018', axlabel='Points', kde=False)
plt.legend()
plt.title('Histogram of top 100 scorers', fontsize=14)