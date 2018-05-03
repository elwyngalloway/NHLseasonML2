#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:23:58 2018

@author: Galloway
"""

import sqlite3
import numpy as np
#import pandas as pd

db_name = "NHLseasonML_seasonstats.db"

# connect to our database that will hold everything, or create it if it doesn't exist
conn = sqlite3.connect(db_name)


#%%
# return all table names from db

with conn:
    # get the cursor so we can do stuff
    cur = conn.cursor()
    
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';") # finds all tables from db    

    #cols = [description[0] for description in cur.description] #records column names for all columns in table s_skater...
    data = cur.fetchall()

    #print("SQLite version: %s" % data)

    print(data)

#%%
# return all column names within a table
with conn:
    # get the cursor so we can do stuff
    cur = conn.cursor()
    
    cur.execute('SELECT * FROM s_skater_summary') #finds column names for all columns in table s_skater...
    
    cols = [description[0] for description in cur.description] #records column names for all columns in table s_skater...
            
cols[:]

#%%
# retrieve data, including playerPositionCode, from s_skater_summary

with conn:
        # get the cursor so we can do stuff
        cur = conn.cursor()

        # extract a number of columns from s_skater_summary
        cur.execute("SELECT seasonId, playerPositionCode, assists, goals, gamesplayed, playerId FROM s_skater_summary")

        data = cur.fetchall()
        
        print(data[:5]) # print 



#%%
















