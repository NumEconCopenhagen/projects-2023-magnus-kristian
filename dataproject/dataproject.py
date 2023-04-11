def keep_regs(df, regs):
    """ Example function. Keep only the subset regs of regions in data.

    Args:
        df (pd.DataFrame): pandas dataframe 

    Returns:
        df (pd.DataFrame): pandas dataframe

    """ 
    
    for r in regs:
        I = df.reg.str.contains(r)
        df = df.loc[I == False] # keep everything else
    
    return df

def import_data(indicator):
    """ Imports data"""
    wb_gdp = wb.download(indicator='SI.POV.GINI', country=['SE','DK','NO'], start=1990, end=2017)

    wb_gdp = wb_gdp.rename(columns = {'SI.POV.GINI':'GINI'})
    wb_gdp = wb_gdp.reset_index()
    wb_gdp.sample(5)