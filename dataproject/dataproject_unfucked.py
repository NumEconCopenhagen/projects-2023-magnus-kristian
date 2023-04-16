#Import packages
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

    # Download data using API
    wb0 = wb.download(country=in_country, indicator=varlist_to_get, start=start_year, end=end_year)

    # Cleans data
    wb1 = (wb0
         .rename(columns = varlist_as_dict)  # Rename data columns
         .reset_index()                      # Change multi-index ['country','year'] to separate columns (because we like our data this way)
         .astype({'year':'int'})             # Change data type of year
         .astype({'country':'string'}))      # Change data type of country

    return wb1

#Function: Imports WGI data
def WGI_DataFrame(file_path):
    '''Downloads and cleans WGI-data from CSV-file'''
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
    dataframe = (dataframe.
                rename(columns=col_dict).                    # Renames WGI-measures
                rename(columns={'Country Name': 'country'})) #Renames country var

    # Using the 'wide_to_long()' method from Pandas, the wb_wgi DataFrame is converted from a wide format to a long format, with the new column 'year' created for the year data is collected.
    dataframe = pd.wide_to_long(dataframe, stubnames='wgi', i=['country','Series Name'], j='year')

    # Resetting the index of the wb_wgi DataFrame with the reset_index() method.
    dataframe = dataframe.reset_index() 

    # Renaming the 'Series Name' column to 'ser' using the rename() method.
    dataframe.rename(columns = {'Series Name':'ser'}, inplace=True)

    #Gets a list of all the series in the data
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

#Function: Merges datasets
def merge(wb_dataset, wgi_dataset):
    '''Merges the datasets from the API and from the .csv-file.
    It also sets empty values equal to nan and changes the .type of variables to be .float'''

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

#Function: Creates correlation graphs across time
def corr(df,var1,var2):
    '''Creates a figure with 4 panels, where one variable is regressed on the other, which is lagged 0, 10, 20 and 30 years.''' 
    # Drops missing values from final dataframe based on the var1 and var2 columns and creates a new dataframe called final_data2.
    final_data2 = df.copy().dropna(subset=[var1, var2], how='any')

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))

    #Creates a new label for the lagged variables
    var1_lagged = var1 + '_lagged' 

#Function: Creates a scatter plot
def ScatterPlot(df,varx,vary,varcolor):
    '''Creates a scatter plot where the correlation between vary and varx is shown in a given year, the dots are colored based on their value of varcolor'''
    # We create a new DataFrame called final_data by dropping all rows with missing values in varx, vary and carcolor columns using the dropna() method.
    final_data = df.dropna(subset=[varx, vary, varcolor], how='any')

    # We define variables cmin and cmax, which respectively store the minimum and maximum values of the varcolor column in the final_data DataFrame. These values are used to set the range of the colorbar in the plot.
    cmin = final_data[varcolor].min()
    cmax = final_data[varcolor].max()

    # We define a function called plot_scatter that creates a scatter plot of vary vs varx for a given year. The plot is colored based on the varcolor variable, which is represented by a colorbar.
    def plot_scatter(year):
        fig, ax = plt.subplots()
        scatter = ax.scatter(final_data[final_data['year'] == year][varx], final_data[final_data['year'] == year][vary],
                            c=final_data[final_data['year'] == year][varcolor], alpha=0.5, vmin=cmin, vmax=cmax)
        ax.set_xlabel(r'$GINI$', fontsize=15)
        ax.set_ylabel(r'$GDP$', fontsize=15)
        ax.set_title(f'{varx} and {vary} (Year: {year})')
        ax.set_xlim(final_data[varx].min(), final_data[varx].max())  # Set x-axis limits
        ax.set_ylim(final_data[vary].min(), final_data[vary].max())  # Set y-axis limits
        ax.grid(True)
        fig.tight_layout()
        # Add color bar
        cbar = plt.colorbar(scatter)
        cbar.set_label(varcolor, fontsize=12)  # Set color bar label
        plt.show()

    # Create a list of unique years in ascending order
    years = sorted(final_data['year'].unique())  

    # We create an interactive plot of the plot_scatter function, with the year variable being controlled by a slider. The slider allows to choose a specific year to plot the scatter plot for.
    interact(plot_scatter, year=years)

# Function: Creates correlation graphs across time
def corr(df,var1,var2):
    '''Creates a figure with 4 panels, where one variable is regressed on the other, which is lagged 0, 10, 20 and 30 years.''' 
    # Drops missing values from final dataframe based on the var1 and var2 columns and creates a new dataframe called final_data2.
    final_data2 = df.copy().dropna(subset=[var1, var2], how='any')

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))

    # Creates a new label for the lagged variables
    var1_lagged = var1 + '_lagged' 

    # Function: Runs for each panel 
    for i, ax in enumerate(axs.flatten()):
        ''' Loop through each subplot and plot the scatterplot with appropriate GINI lag'''
        # Lag GINI by i years
        final_data2[var1_lagged] = final_data2[var1].shift(i * 20)
        
        # Drops missing values based on the var1_lagged and var2 columns from the final_data2 dataframe.
        final_data2_filtered = final_data2.dropna(subset=[var1_lagged, var2], how='any')
        
        # Plots a scatterplot using sns.histplot() function with var1_lagged on the x-axis, var2 on the y-axis, and the final_data2_filtered dataframe
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
        
        # Adds regression line to the scatterplot
        x = np.linspace(final_data2[var1_lagged].min(), final_data2[var1_lagged].max(), 200)
        y = slope * x + intercept
        ax.plot(x, y, color='black', linewidth=1)
        
        # Put result of regression in the corner of each panel
        ax.annotate(f'Slope: {slope:.2f}{significance_slope}\nIntercept: {intercept:.2f}{significance_intercept}', xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top')
        
        ax.set_xlabel(f'{var1} lagged {i * 20} years', fontsize=12)  # x-axis label with lag
        ax.set_ylabel(var2, fontsize=12)  # y-axis label
        ax.set_title(f'Panel {i + 1}', fontsize=12)  # plot title
        ax.grid(True)
    # Calls the plt.tight_layout() function to optimize the layout of the subplots and displays the subplots using the plt.show() function.
    plt.tight_layout()
    plt.show

#Function: Makes regressions
def regression(df, y_var, x_list):
    '''Regresses yvar on x_list'''
    # The X variable is created by selecting the columns of the final DataFrame that correspond to the independent variables using the loc method.
    X = df.loc[:,x_list]
    
    # A loop is used to shift the values of each independent variable backwards by 10 time years using the shift method.
    for i in x_list:
        X[i] = X[i].shift(-10) 

    # The X variable is then augmented with a constant term using the add_constant function from the statsmodels.api module
    X = sm.add_constant(X)

    # The dependent variable y is selected from the final DataFrame using the loc method.
    y = df.loc[:,y_var]

    # An OLS regression model is created using the OLS function from the statsmodels.api module, with y as the dependent variable and X as the independent variables. The missing='drop' argument specifies that any rows with missing values should be dropped.
    model = sm.OLS(y,X, missing='drop')

    # The fit method is used to fit the model and store the results in the result variable.
    result = model.fit()
        
    print(result.summary())

#Function: Regression of sd. of vary-detrended on x_list vars. 
def regression1(df, vary, x_list, years, year):
    '''Runs a regression which checks if the standard deviation of vary around tis trend is larger for countries with bad WGI-values'''
    # sort dataset to contain specicic years.
    df1 = df.loc[df['year'].isin(years)]
    df1 = df1.dropna(subset=[vary], how='any')

    # detrend 'vary'
    y = df1.loc[:, vary]
    df1['y_detrended'] = signal.detrend(y, type='linear')

    #Computes standard deviation of detrended variable.
    df1['std'] = df1.groupby('country')['y_detrended'].transform('std')

    # Select only one year
    df2 = df1.loc[df1['year'] == year]

    # Regress std. on lagged governance
    X = df2.loc[:, x_list]  # COC should be changed to governance indicator
    X = sm.add_constant(X)

    y = df2.loc[:, 'std']
    # An OLS regression model is created using the OLS function from the statsmodels.api module, with y as the dependent variable and X as the independent variables. The missing='drop' argument specifies that any rows with missing values should be dropped.
    model = sm.OLS(y,X, missing='drop')

    # The fit method is used to fit the model and store the results in the result variable.
    result = model.fit()
        
    print(result.summary())
