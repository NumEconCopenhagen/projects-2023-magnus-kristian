import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from pandas_datareader import wb
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact
import seaborn as sns
from scipy.stats import linregress
import statsmodels.api as sm 
from scipy import signal

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
         .reset_index()                      # Change multi-index ['country','year'] to separate columns (because we like our data this way)
         .astype({'year':'int'})             # Change data type of year
         .astype({'country':'string'}))      # Change data type of country

    return wb1

def WGI_DataFrame(file_path):
#   def __init__(file_path, file_path2, in_country, varlist_as_dict, start_year, end_year):
    dataframe = pd.read_csv(file_path)

    # list of columns to drop
    drop_these = ['Country Code', 'Series Code']
    # drop columns
    dataframe.drop(drop_these, axis=1, inplace=True)

    # dictionary to map column names to new names
    col_dict = {}
    for i in range(1996, 2022):
        col_dict[str(i)+" [YR"+str(i)+"]"] = f'wgi{i}'
    
    # rename columns
    dataframe.rename(columns=col_dict, inplace=True)
    dataframe.rename(columns={'Country Name': 'country'}, inplace=True)

    # Using the 'wide_to_long()' method from Pandas, the wb_wgi DataFrame is converted from a wide format to a long format, with the new column 'year' created for the year data is collected.
    dataframe = pd.wide_to_long(dataframe, stubnames='wgi', i=['country','Series Name'], j='year')

    # Resetting the index of the wb_wgi DataFrame with the reset_index() method.
    dataframe = dataframe.reset_index() 

    # Renaming the 'Series Name' column to 'ser' using the rename() method.
    dataframe.rename(columns = {'Series Name':'ser'}, inplace=True)

    #Get a list of all the series in the data
    namelist = dataframe.ser.unique()

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
        dataframe.loc[dataframe.ser == name, 'ser'] = newnames[name]

    # Creating a new list of series names with the new names, using the unique() method on the ser column.
    newnamelist = dataframe.ser.unique()

    # We create a pivot table with wb_wgi dataframe, where the index is set to ['country', 'year'], the columns are set to ser, and the values are set to wgi. This is stored in a new dataframe called wb_new.
    dataframe=pd.pivot(dataframe, index=['country','year'], columns = 'ser', values= 'wgi')
        
    # We reset the index of 'wb_new'
    dataframe=dataframe.reset_index()

    return dataframe

def merge(wb_dataset, wgi_dataset):
    # We then merge the self.dataframe and wb_new dataframes on the columns 'year' and 'country', using an outer join.
    final = pd.merge(wb_dataset, wgi_dataset, on=['year', 'country'], how = 'outer')
    
    # We create a list of column names called 'col_list' which includes the names of columns to be processed in the loop
    col_list = ['COC', 'GOV', 'REQ', 'ROL', 'RSA', 'VOA']

    # We use the loc function to locate all rows in final dataframe where the value of column i (where i is each column name in col_list) is equal to "..", and replaces those values with NaN. This process converts the string data type to float data type, which is easier to work with for numeric analysis.
    for i in col_list:
        final.loc[final[i]==".."] = np.nan
        final[i]=final[i].astype(float)
        final = final.dropna(subset=['country','year'], how='any')
        final.year = final.year.astype(int)

    return final

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

def corr(df,var1,var2):
    # Drops missing values from final dataframe based on the GINI and GDP_growth columns and creates a new dataframe called final_data2.
    final_data2 = df.copy().dropna(subset=[var1, var2], how='any')

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))

    var1_lagged = var1 + '_lagged' 

    # Loop through each subplot and plot the scatterplot with appropriate GINI lag
    for i, ax in enumerate(axs.flatten()):

        # Lag GINI by i years
        final_data2[var1_lagged] = final_data2[var1].shift(i * 10)
        
        # Drops missing values based on the GINI_lagged and GDP_growth columns from the final_data2 dataframe.
        final_data2_filtered = final_data2.dropna(subset=[var1_lagged, var2], how='any')
        
        # Plots a scatterplot using sns.histplot() function with GINI_lagged on the x-axis, GDP_growth on the y-axis, and the final_data2_filtered dataframe
        sns.histplot(x=var1, y=var2, data=final_data2_filtered, ax=ax, bins=30, cbar=True, cmap='viridis', cbar_kws={'label': 'Count'})
        
        # Add regression line and display coefficients with significance stars
        slope, intercept, r_value, p_value, std_err = linregress(final_data2_filtered[var1_lagged], final_data2_filtered[var2])
        significance_slope = ""
        significance_intercept = ""
        if p_value < 0.001:
            significance_slope = "***"
            significance_intercept = "***"
        elif p_value < 0.01:
            significance_slope = "**"
            significance_intercept = "**"
        elif p_value < 0.05:
            significance_slope = "*"
            significance_intercept = "*"
        
        # Add regression line to the scatterplot
        x = np.linspace(final_data2_filtered[var1_lagged].min(), final_data2_filtered[var2].max(), 100)
        y = slope * x + intercept
        ax.plot(x, y, color='black', linewidth=1)
        
        ax.annotate(f'Slope: {slope:.2f}{significance_slope}\nIntercept: {intercept:.2f}{significance_intercept}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top')
        
        ax.set_xlabel(f'{var1} lagged {i * 10} years', fontsize=12)  # x-axis label with lag
        ax.set_ylabel(var2, fontsize=12)  # y-axis label
        ax.set_title(f'Panel {i + 1}', fontsize=12)  # plot title
        ax.grid(True)
    # Calls the plt.tight_layout() function to optimize the layout of the subplots and displays the subplots using the plt.show() function.
    plt.tight_layout()
    plt.show()

# define function to do regression: 
def regression(df, y_var, x_list):

    # The X variable is created by selecting the columns of the final DataFrame that correspond to the six independent variables using the loc method.
    X = df.loc[:,x_list]
    
    # A loop is used to shift the values of each independent variable forward by 10 time years using the shift method. This is done in order to lag the variable by 10 years. 
    for i in x_list:
        X[i] = X[i].shift(-10) 

    # The X variable is then augmented with a constant term using the add_constant function from the statsmodels.api modul
    X = sm.add_constant(X)

    # The dependent variable y is selected from the final DataFrame using the loc method.
    y = df.loc[:,y_var]

    # An OLS regression model is created using the OLS function from the statsmodels.api module, with y as the dependent variable and X as the independent variables. The missing='drop' argument specifies that any rows with missing values should be dropped.
    model = sm.OLS(y,X, missing='drop')

    # The fit method is used to fit the model and store the results in the result variable.
    result = model.fit()
        
    print(result.summary())


#Skal debugges!
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

