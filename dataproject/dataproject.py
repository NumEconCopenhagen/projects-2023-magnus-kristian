import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from pandas_datareader import wb


class WGI_DataFrame:
    def __init__(self, file_path):
        self.dataframe = pd.read_csv(file_path)

        # list of columns to drop
        self.drop_these = ['Country Code', 'Series Code']

        # dictionary to map column names to new names
        self.col_dict = {}
        for i in range(1996, 2022):
            self.col_dict[str(i)+" [YR"+str(i)+"]"] = f'wgi{i}'

        # process dataframe
        self.__process_dataframe()

    def __process_dataframe(self):
        # drop columns
        self.dataframe.drop(self.drop_these, axis=1, inplace=True)

        # rename columns
        self.dataframe.rename(columns=self.col_dict, inplace=True)
        self.dataframe.rename(columns={'Country Name': 'country'}, inplace=True)

        # Using the 'wide_to_long()' method from Pandas, the wb_wgi DataFrame is converted from a wide format to a long format, with the new column 'year' created for the year data is collected.
        self.dataframe = pd.wide_to_long(self.dataframe, stubnames='wgi', i=['country','Series Name'], j='year')

        # Resetting the index of the wb_wgi DataFrame with the reset_index() method.
        self.dataframe = self.dataframe.reset_index() 

        # Renaming the 'Series Name' column to 'ser' using the rename() method.
        self.dataframe.rename(columns = {'Series Name':'ser'}, inplace=True)

        #Get a list of all the series in the data
        namelist = self.dataframe.ser.unique()

        # Defining a dictionary called newnames that maps the long series names to short names.
        newnames = {'Control of Corruption: Estimate':'COC',
                    'Government Effectiveness: Estimate':'GOV', 
                    'Political Stability and Absence of Violence/Terrorism: Estimate':'RSA', 
                    'Regulatory Quality: Estimate':'REQ', 
                    'Rule of Law: Estimate':'ROL', 
                    'Voice and Accountability: Estimate':'VOA'}

        # Renaming the 'ser' column with the new names based on the newnames dictionary, using a loop that iterates through each series name in the DataFrame.
        for index, name in enumerate(namelist):
            print(index, "Now rename Series", name, "=", newnames[name])
            self.dataframe.loc[self.dataframe.ser == name, 'ser'] = newnames[name]

        # Creating a new list of series names with the new names, using the unique() method on the ser column.
        newnamelist = self.dataframe.ser.unique()

        # We create a pivot table with wb_wgi dataframe, where the index is set to ['country', 'year'], the columns are set to ser, and the values are set to wgi. This is stored in a new dataframe called wb_new.
        wb_new=pd.pivot(self.dataframe, index=['country','year'], columns = 'ser', values= 'wgi')
        
        # We reset the index of 'wb_new'
        wb_new=wb_new.reset_index()
        
        # We then merge the self.dataframe and wb_new dataframes on the columns 'year' and 'country', using an outer join.
        self.final = pd.merge(self.dataframe, wb_new, on=['year', 'country'], how = 'outer')
        
        # We create a list of column names called 'col_list' which includes the names of columns to be processed in the loop
        col_list = ['COC', 'GOV', 'REQ', 'ROL', 'RSA', 'VOA']

        # We use the loc function to locate all rows in final dataframe where the value of column i (where i is each column name in col_list) is equal to "..", and replaces those values with NaN. This process converts the string data type to float data type, which is easier to work with for numeric analysis.
        for i in col_list:
            self.final.loc[self.final[i]==".."] = np.nan
    
    def final_dataframe(self):
        return self.final
    

import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact

class ScatterPlot:
    def __init__(self, data):
        self.final_data = data.dropna(subset=['GINI', 'GDP', 'COC'], how='any')
        self.quality = self.final_data.COC
        self.cmin = self.final_data.COC.min()
        self.cmax = self.final_data.COC.max()
        self.years = sorted(self.final_data['year'].unique())

    def plot_scatter(self, year):
        fig, ax = plt.subplots()
        scatter = ax.scatter(self.final_data[self.final_data['year'] == year]['GINI'], self.final_data[self.final_data['year'] == year]['GDP'],
                             c=self.final_data[self.final_data['year'] == year]['COC'], alpha=0.5, vmin=self.cmin, vmax=self.cmax)
        ax.set_xlabel(r'$GINI$', fontsize=15)
        ax.set_ylabel(r'$GDP$', fontsize=15)
        ax.set_title('GINI and GDP (Year: {})'.format(year))
        ax.set_xlim(self.final_data['GINI'].min(), self.final_data['GINI'].max())  # Set x-axis limits
        ax.set_ylim(self.final_data['GDP'].min(), self.final_data['GDP'].max())  # Set y-axis limits
        ax.grid(True)
        fig.tight_layout()
        # Add color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label('COC', fontsize=12)  # Set color bar label
        plt.show()

    def interact_plot(self):
        interact(self.plot_scatter, year=self.years)
     
     


# Function: Download World Bank data

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



# Can the following be deleted...?

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


