"""Clean and reshape incoming data"""

import pandas as pd

def create_demand_df(df):
    # convert date to pandas datetime object
    df["date_time"] = pd.to_datetime(df['date_time'], format='%Y-%m-%d %H:%M:%S')

    # create separate date and hour columns to aid with pivoting
    df["date"] = df["date_time"].dt.date  # Add separate date string 
    df["hour"] = df["date_time"].dt.hour  # Add separate hour string

    # Drop all columns except "Cleaned Demand (MW)"
    df_dropped = df.drop(["raw demand (MW)", "category", "forecast demand (MW)", "date_time"], axis=1)
    df_demand = df_dropped.pivot(index = "date", columns = "hour", values = "cleaned demand (MW)")

    # return the pivoted df
    return df_demand
