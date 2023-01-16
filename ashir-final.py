

# accessing the important libraries for data processing
import pandas as pd
import numpy as np

#libraries for classification
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import sklearn.preprocessing as prep
import scipy.optimize as opt
from errors import err_ranges

#libraries for visualization
import seaborn as sb
import matplotlib.pyplot as plt



def read_and_transpose_worldbank_format_file(filename):
    ''' 
    it reads a file in world bank format and returns its transpose

    Parameters:
        filename - the path of the file
    Returns:
        data - the data after reading the file
        data_t - transposed dataset with countries as columns
    '''
    
    # reads the data in a data frame
    data = pd.read_csv(filename, skiprows=4)

    # Removes the Extra Column
    data = data.drop([ "Unnamed: 66"], axis=1)

    # Transpose the DataFrame so that it has countries as its columns
    data_t = data.T.copy()

    # Return both the Data frames
    return data, data_t


def extract_indicator_data(df, indicator):
    '''
    pre-process the indicator data frame by removing the unnecessary columns and rows
    '''
    df = df.loc[df["Indicator Code"] == indicator, :]
    indicator_name= df['Indicator Name'].iloc[0]
    #indicator_name = indicator_name.iloc[0,0]
    df.index = df.iloc[:, 0] # making the country name as the index column
    
    df = df.drop(["Country Name", "Indicator Name", "Country Code", "Indicator Code"], axis=1)

    return df, indicator_name



def plot_line_graph(df, title='', xlbl = '', ylbl = ''):
    '''
    plots a line graph with data frame df
    '''
    plt.plot(df)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.xticks([x for x in np.array(df.index) if int(x) %5 == 0], rotation=90)
    plt.title(title)
    plt.show()



def combine_dfs(df1, df_gdp,yr):
    '''
    this function combines two dataframes into a single dataframe with two columns

    '''
    df = df_gdp[yr].copy()
    df['col'] = df1[yr].to_numpy()
    df.rename(columns = {yr[0]:'GDP'}, inplace = True)
    df.dropna(inplace=True)


    df.insert(0, 'col', df.pop('col'))
    
    return df

def kmeans_clustering(data,nclusters):
    '''
    creates k-meanss clustering and fits the model
    '''
    kmeans = cluster.KMeans(n_clusters=nclusters, n_init=10);
    kmeans.fit(data) 
    return kmeans



def draw_all_years_kmeans_plots(data, gdp_data, title="", xlabel="", ylabel="", xscale='linear', yscale='linear'):
    '''
    this function draws year by year k-means plot of the two input dataframes
    '''
    clusters_n = 4
    for i in range(1960,2021,10):
        yr = str(i)
        df_year = combine_dfs(data, gdp_data, [yr])
        
        if df_year.shape[0] != 0:

            # Training and Fitting the data on Kmeans with 2 clusters
            kmeans = kmeans_clustering(df_year, clusters_n)
            labels = kmeans.labels_ 

            cen = kmeans.cluster_centers_

            silhoutte = skmet.silhouette_score(df_year, labels)

            # plot using the labels to select colour
            
            col_names=df_year.columns

            for l in range(clusters_n): # 
                plt.plot(df_year[labels==l][col_names[0]], df_year[labels==l][col_names[1]], "o", markersize=3)
            #

            # show cluster centres
            for ic in range(clusters_n):
                xc, yc = cen[ic,:]
                plt.plot(xc, yc, "^y", markersize=10)

            plt.xlabel(xlabel=xlabel)
            plt.ylabel(ylabel)
            plt.yscale(xscale)
            plt.xscale(yscale)
            plt.title(title+ "Year: " + yr)
            
            plt.show()


# same function as in lecture
def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


def logistic_curve_fit(dfdata, title='World', xlabel='', ylabel=''):
    '''
    this function plots the logistics curve fit
    '''
    # converting Year from string to numeric 
    dfdata["Year"] = pd.to_numeric(dfdata["Year"])

    # fitting 
    param, covar = opt.curve_fit(logistic, dfdata.iloc[:,0], dfdata.iloc[:,1], p0=(np.mean(dfdata.iloc[:,1]), 0.03, 1900.0))
    sigma = np.sqrt(np.diag(covar))

    arr = dfdata['Year'].to_numpy()
    years = np.arange(min(arr), max(arr) + 10)

    low, up = err_ranges(years, logistic, param, sigma)
    plt.fill_between(years, low, up, color="yellow", alpha=0.7)

    forecast = logistic(years, *param)

    dfdata["fit"] = logistic(dfdata["Year"], *param)

    plt.plot(dfdata['Year'],dfdata[[dfdata.columns[1], 'fit']])
    plt.plot(years, forecast)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()




if __name__ == '__main__':

    #reading the worldbank dataset
    df, df_t= read_and_transpose_worldbank_format_file('world_bank_data.csv')


    df_extreme_weather, title_extreme_weather = extract_indicator_data(df, 'EN.CLC.MDAT.ZS')
    df_greenhouse_gas, title_greenhouse_gas = extract_indicator_data(df, 'EN.ATM.GHGT.ZG')
    df_urban_population, title_urban_population = extract_indicator_data(df, 'SP.URB.TOTL.IN.ZS')
    df_co2_emission, title_co2_emission = extract_indicator_data(df, 'EN.ATM.CO2E.PC')


    # Reading and pre-processing the GDP dataset
    df_gdp = pd.read_csv('gdp_per_capita.csv', skiprows=4, index_col=0) 
    title_gdp = df_gdp['Indicator Name'].iloc[0] # Extract the name of the indicator
    df_gdp.drop(df_gdp.columns[[0,1,2,-1]], axis=1, inplace=True) # drop all the unnecessary columns


    #calculating the world GDP
    world_gdp = df_gdp.sum(axis=0)
    world_gdp.plot(title="World GDP", xlabel='Year', ylabel='World GDP per Capita');
    plt.show()

    #plotting the urban population graph
    plot_line_graph(df_urban_population.sum(axis=0) , "Urban Population", "Year", "Population Growth")
    dff = df_co2_emission.sum(axis=0)
    dff = dff.loc[dff != 0] # remove the rows with zero value
    plot_line_graph(dff, title_co2_emission, "Year", "CO2 Emission")
    plot_line_graph(df_co2_emission.sum(axis=0), title_co2_emission, "Year", "CO2 Emission")


    #Training and Plotting the K-Means Clustering
    draw_all_years_kmeans_plots(df_co2_emission, df_gdp, title="K-Means Clustering ", xlabel = "CO2 Emission", ylabel="GDP per capita", yscale='log')
    draw_all_years_kmeans_plots(df_urban_population, df_co2_emission, title="K-Means Clustering ", xlabel=title_urban_population, ylabel=title_co2_emission)


    #Plotting the curve_fit
    df_world_gdp = pd.DataFrame({"Year": world_gdp.index.to_numpy(), "World": world_gdp}).reset_index().iloc[:, 1:]
    logistic_curve_fit(df_world_gdp, "World GDP Per capita", xlabel="Years", ylabel="GDP per capita")

    df_pop = df_urban_population.T.sum(axis=1).reset_index()
    df_pop.columns = ["Year", "CO2_Emission"]
    df_pop = df_pop.loc[df_pop.iloc[:,1] != 0] # remove the zeros

    logistic_curve_fit(df_pop, "Logistic Curve Fit of Urban Population", xlabel="Years", ylabel='Urban Population')
