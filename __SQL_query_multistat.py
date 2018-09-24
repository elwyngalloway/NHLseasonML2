#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 20:53:23 2018

@author: Galloway
"""

db_name = "NHLseasonML_seasonstats.db"
conn = sqlite3.connect(db_name)

with conn:
    cur = conn.cursor()
    
    cur.execute("SELECT DISTINCT s_skater_summary.seasonId, \
                s_skater_summary.playerId, s_bio_info.playerBirthDate, \
                s_bio_info.playerDraftOverallPickNo, s_skater_summary.playerPositionCode, \
                s_skater_summary.shots, s_skater_summary.timeOnIcePerGame, \
                s_skater_summary.gamesplayed, \
                s_skater_summary.plusMinus, \
                s_time_on_ice.ppTimeOnIcePerGame, s_time_on_ice.shTimeOnIcePerGame, \
                s_time_on_ice.evTimeOnIcePerGame, \
                s_skater_points.evAssists, s_skater_points.ppAssists, \
                s_skater_goals.enGoals, s_skater_goals.ppGoals, s_skater_goals.evGoals, \
                s_realtime_events.blockedShots, s_realtime_events.hitsPerGame \
                FROM s_skater_summary \
                INNER JOIN s_bio_info \
                    ON s_bio_info.playerId = s_skater_summary.playerId \
                    AND s_bio_info.seasonId = s_skater_summary.seasonId \
                INNER JOIN s_time_on_ice \
                    ON s_time_on_ice.playerId = s_skater_summary.playerId \
                    AND s_time_on_ice.seasonId = s_skater_summary.seasonId \
                INNER JOIN s_skater_points \
                    ON s_skater_points.playerId = s_skater_summary.playerId \
                    AND s_skater_points.seasonId = s_skater_summary.seasonId \
                INNER JOIN s_skater_goals \
                    ON s_skater_goals.playerId = s_skater_summary.playerId \
                    AND s_skater_goals.seasonId = s_skater_summary.seasonId \
                INNER JOIN s_realtime_events \
                    ON s_realtime_events.playerId = s_skater_summary.playerId \
                    AND s_realtime_events.seasonId = s_skater_summary.seasonId \
                WHERE s_skater_summary.playerID = ? \
                AND s_skater_summary.seasonId NOT IN (20182019)", [8450725])

    data = cur.fetchall()
    
data[0]
#%% Pretty much frickin' everything!

def extractlagprediction(player, stats4lag, lag ):
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
                s_skater_summary.assists, s_skater_summary.gamesplayed, \
                s_skater_summary.plusMinus, \
                s_time_on_ice.ppTimeOnIcePerGame, s_time_on_ice.shTimeOnIcePerGame, \
                s_time_on_ice.evTimeOnIcePerGame, s_time_on_ice.shifts, \
                s_skater_points.evAssists, s_skater_points.ppAssists, \
                s_skater_goals.enGoals, s_skater_goals.ppGoals, s_skater_goals.evGoals, \
                s_realtime_events.blockedShots, s_realtime_events.hitsPerGame \
                FROM s_skater_summary \
                INNER JOIN s_bio_info \
                    ON s_bio_info.playerId = s_skater_summary.playerId \
                    AND s_bio_info.seasonId = s_skater_summary.seasonId \
                INNER JOIN s_time_on_ice \
                    ON s_time_on_ice.playerId = s_skater_summary.playerId \
                    AND s_time_on_ice.seasonId = s_skater_summary.seasonId \
                INNER JOIN s_skater_points \
                    ON s_skater_points.playerId = s_skater_summary.playerId \
                    AND s_skater_points.seasonId = s_skater_summary.seasonId \
                INNER JOIN s_skater_goals \
                    ON s_skater_goals.playerId = s_skater_summary.playerId \
                    AND s_skater_goals.seasonId = s_skater_summary.seasonId \
                INNER JOIN s_realtime_events \
                    ON s_realtime_events.playerId = s_skater_summary.playerId \
                    AND s_realtime_events.seasonId = s_skater_summary.seasonId \
                WHERE s_skater_summary.playerID = ? \
                AND s_skater_summary.seasonId NOT IN (20182019)", [player])
    
    if len(data) > 0:

        df = pd.DataFrame(data)

        df.columns = ('year', 'playerID', 'birthYear','draftPos', 'position',
                      'points', 'goals', 'ppPoints', 'shots', 'ToipG',
                      'assists', 'games', 'plusMinus', 'ppToipG', 'shToipG',
                      'evToipG', 'shifts', 'evAssists', 'ppAssists', 'enGoals',
                      'ppGoals', 'evGoals', 'blocks', 'hitsPerGame')
        df['birthYear'] = pd.to_datetime(df['birthYear']).dt.year
        df['birthYear'] = df['year'] // 10000 - df['birthYear']
        df = df.rename(index=str, columns={'birthYear': 'age'})
        df['position'] = df['position'].astype('category',categories=['C', 'D', 'L', 'R'])
        df = pd.concat([df,pd.get_dummies(df['position'], prefix='position')],axis=1)
        df.drop(['position'],axis=1, inplace=True)
        df['draftPos'].replace('', 300, inplace=True)
        df = df.sort_values(['year'],ascending = False)
        dfshift = df.shift(lag)
        
        # Add the lagged stats:
        
        for stat in stats4lag:
            
            # name the columns of the shifted df
            dfshift = dfshift.rename(index=str, columns={stat : str(stat + 'lag')})
            
            # find the index of the column desired for lagging
            columnindex = df.columns.get_loc(stat)

            # append the appropriate column of the shifted df to the end of the original df
            df = df.join(dfshift.iloc[:,columnindex]).iloc[lag-1:,:]  # This line is distinct from the extractlag
        
        return np.array(df)

    
    else:
        temp = np.empty((1,16))
        temp.fill(np.nan)
        df = pd.DataFrame(temp)
        df.columns = ('year', 'playerID', 'age','draftPos', 'points', 'goals', 'ppPoints', 'shots', 'timeOnIcePerGame', 'assists', 'games', 'position_C', 'position_D', 'position_L', 'position_R', str(stat4lag + 'lag'))
        
        return np.array(df)







