# def keep_regs(df, regs):
#     """ Example function. Keep only the subset regs of regions in data.

#     Args:
#         df (pd.DataFrame): pandas dataframe 

#     Returns:
#         df (pd.DataFrame): pandas dataframe

#     """ 
    
#     for r in regs:
#         I = df.reg.str.contains(r)
#         df = df.loc[I == False] # keep everything else
    
#     return df

# def import_data(indicator):
#     """ Imports data"""
#     wb_gdp = wb.download(indicator='SI.POV.GINI', country=['SE','DK','NO'], start=1990, end=2017)

#     wb_gdp = wb_gdp.rename(columns = {'SI.POV.GINI':'GINI'})
#     wb_gdp = wb_gdp.reset_index()
#     wb_gdp.sample(5)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from pandas_datareader import wb

def my_wb_downloader(in_country, varlist_as_dict, start_year, end_year):
    '''Downloads and cleans World Bank data in one step'''

    # Get variable names from dictionary used to rename variables
    varlist_to_get = list(varlist_as_dict.keys())

    # Download API
    wb0 = wb.download(country=in_country, indicator=varlist_to_get, start=start_year, end=end_year)

    # Clean data
    wb1 = (wb0
         .rename(columns = varlist_as_dict)  # Rename data columns
         .reset_index()                      # Change multi-index ['country','year'] to separat columns (because we like our data this way)
         .astype({'year':'int'})             # Change data type of year
         .astype({'country':'string'}))      # Change data type of country

    return wb1



# Function: Nomalize a list of variables by group in DataFrame
def standardize_by_group(df, varlist, grouplist):
    '''Normalize variables by mean and standard deviation'''

    # for all variables in varlist
    for var in varlist:
        
        # Generate a new variables *_norm
        new_var = var + '_norm'

        # Subtract the group mean and divide by group std
        df[new_var] = (df[var] - df.groupby(grouplist)[var].transform('mean')) / df.groupby(grouplist)[var].transform('std')

    # Return the modified dataframe
    return df

# Function: Calculate simple statistics for all variables in DataFrame
def calc_simplestats(df, varlist, group_by_year = True, groups_to_print=[]):
    '''Generate new dataframe with descriptive statistics for all variables in varlist for. Leave groups_to_print empty if group_by_year is False'''

    if group_by_year == True:

        assert len(groups_to_print) > 0 # must select some years

        # new dataframe
        df_stats = (df.query('year in @groups_to_print')     # Report only specific years
                  .groupby('year')[varlist]                       # Group by (default is year)
                  .agg(['count','mean', 'std']))                  # Report n, mean, std
    
    if group_by_year == False:

        # new dataframe
        df_stats = (df[varlist].agg(['count','mean', 'std']))                  # Report n, mean, std

    return df_stats
