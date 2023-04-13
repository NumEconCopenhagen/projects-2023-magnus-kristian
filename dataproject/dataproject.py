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