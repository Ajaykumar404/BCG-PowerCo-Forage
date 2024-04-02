#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis
# 
# 

# ### 1)Importing packages
# ### 2)Loading Data with Pandas
# ### 3)Descriptive Statistics of Data
# ### 4)Data Visualization

# ## Importing Packages

# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Shows plots in jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Set plot style
sns.set(color_codes=True)


# In[54]:


pd.set_option('display.max_columns',None)
pd.set_option("display.max_rows",None)


# ---
# 
# ## Loading data with Pandas
# 
# We need to load `client_data.csv` and `price_data.csv` into individual dataframes so that we can work with them in Python. For this notebook and all further notebooks, it will be assumed that the CSV files will the placed in the same file location as the notebook. If they are not, please adjust the directory within the `read_csv` method accordingly.

# In[4]:


client_df = pd.read_csv('./client_data.csv')
price_df = pd.read_csv('./price_data.csv')


# we can look at view the first 5 rows of a dataframe using the `head` method. Similarly, if you wanted to see the last 5, we can use `tail()`

# In[7]:


client_df.head()


# In[16]:


client_df.shape


# In the client data,Most of the columns are numerical but a few are categorical.so we need to transform them into numerical 
# before modeling.

# In[9]:


price_df.head()


# In[17]:


price_df.shape


# In the price Data,we have 8 columns here all of them are numerical.but a lot of zeros

# ---
# 
# ## Descriptive statistics of data
# 
# ### Data types
# 
# It is useful to first understand the data that you're dealing with along with the data types of each column. The data types may dictate how you transform and engineer features.
# 
# To get an overview of the data types within a data frame, use the `info()` method.

# In[11]:


client_df.info()


# In[20]:


price_df.info()


# #Here we can observe that date time columns are not in Datetime format,they need to be converted later.

# ### Statistics
# 
# Now let's look at some statistics about the datasets. We can do this by using the `describe()` method.

# In[7]:


client_df.describe()


# The describe method gives us a lot of information about the client data. The key point to take away from this is that we have highly skewed data, as exhibited by the percentile values.

# In[23]:


price_df.describe()


# over all price data looks good.

# ---
# 
# ## Data visualization
# 
# If you're working in Python, two of the most popular packages for visualization are `matplotlib` and `seaborn`. We highly recommend you use these, or at least be familiar with them because they are ubiquitous!
# 
# Below are some functions that you can use to get started with visualizations. 

# In[33]:


def plot_stacked_bars(dataframe, title_, size_=(18, 10), rot_=0, legend_="upper right"):
    """
    Plot stacked bars with annotations
    """
    ax = dataframe.plot(
        kind="bar",
        stacked=True,
        figsize=size_,
        rot=rot_,
        title=title_
    )

    # Annotate bars
    annotate_stacked_bars(ax, textsize=14)
    # Rename legend
    plt.legend(["Retention", "Churn"], loc=legend_)
    # Labels
    plt.ylabel("Company base (%)")
    plt.show()

def annotate_stacked_bars(ax, pad=0.99, colour="white", textsize=13):
    """
    Add value annotations to the bars
    """

    # Iterate over the plotted rectanges/bars
    for p in ax.patches:
        
        # Calculate annotation
        value = str(round(p.get_height(),1))
        # If value is 0 do not annotate
        if value == '0.0':
            continue
        ax.annotate(
            value,
            ((p.get_x()+ p.get_width()/2)*pad-0.05, (p.get_y()+p.get_height()/2)*pad),
            color=colour,
            size=textsize
        )


# Thhe first function `plot_stacked_bars` is used to plot a stacked bar chart. An example of how you could use this is shown below:

# In[34]:


churn = client_df[['id', 'churn']]
churn.columns = ['Companies', 'churn']
churn_total = churn.groupby(churn['churn']).count()
churn_percentage = churn_total / churn_total.sum() * 100
plot_stacked_bars(churn_percentage.transpose(), "Churning status", (5, 5), legend_="lower right")


# Around 10% of the customers churned.

# ## Sales Channel

# In[29]:


channel=client_df[["id","channel_sales","churn"]]
channel=channel.groupby([channel["channel_sales"],channel["churn"]])["id"].count().unstack(level=1).fillna(0)
channel_churn = (channel.div(channel.sum(axis=1), axis=0) * 100).sort_values(by=[1], ascending=False)


# In[32]:


plot_stacked_bars(channel_churn,"sales_channel",rot_=30)


# Interestingly,the churning columns are distributed over the 5 values in 'channel_sales'. As well as this, the value of `MISSING` has a churn rate of 7.6%. `MISSING` indicates a missing value and was added by the team when they were cleaning the dataset. This feature could be an important feature when it comes to building our model.
# 
# ### Consumption
# 
# Let's see the distribution of the consumption in the last year and month. Since the consumption data is univariate, let's use histograms to visualize their distribution.

# In[35]:


def plot_distribution(dataframe, column, ax, bins_=50):
    """
    Plot variable distirbution in a stacked histogram of churned or retained company
    """
    # Create a temporal dataframe with the data to be plot
    temp = pd.DataFrame({"Retention": dataframe[dataframe["churn"]==0][column],
    "Churn":dataframe[dataframe["churn"]==1][column]})
    # Plot the histogram
    temp[["Retention","Churn"]].plot(kind='hist', bins=bins_, ax=ax, stacked=True)
    # X-axis label
    ax.set_xlabel(column)
    # Change the x-axis to plain style
    ax.ticklabel_format(style='plain', axis='x')


# In[36]:


consumption = client_df[['id', 'cons_12m', 'cons_gas_12m', 'cons_last_month', 'imp_cons', 'has_gas', 'churn']]

fig, axs = plt.subplots(nrows=1, figsize=(18, 5))

plot_distribution(consumption, 'cons_12m', axs)


# In[37]:


fig, axs = plt.subplots(nrows=4, figsize=(18, 25))

plot_distribution(consumption, 'cons_12m', axs[0])
plot_distribution(consumption[consumption['has_gas'] == 't'], 'cons_gas_12m', axs[1])
plot_distribution(consumption, 'cons_last_month', axs[2])
plot_distribution(consumption, 'imp_cons', axs[3])


# Here the consumption data is highly positive skewed, presenting a very long-right tail towards the higher values of the distribution.the distribution in the  higher and lower end are likely to be outliers.we can use boxplot to get standardized
# way to identify the outliers present in the data. a box plot displays the distribution based on the five number summary:
# -Minimum
# -First quartile(Q1)
# -median 
# -Third quartile(Q2)
# -Maximum
# 
# it can reveal outliers and what their values are.it can also tell is our data symmetrical,how tightly our data is grouped and if/how our data skewed.

# In[50]:


fig, axs = plt.subplots(nrows=4, figsize=(18,25))

# Plot histogram
sns.boxplot(consumption["cons_12m"], ax=axs[0])
sns.boxplot(consumption[consumption["has_gas"] == "t"]["cons_gas_12m"], ax=axs[1])
sns.boxplot(consumption["cons_last_month"], ax=axs[2])
sns.boxplot(consumption["imp_cons"], ax=axs[3])

# Remove scientific notation
plt.show()


# we deal with skewness in the feature engineering

# ## Forecast

# In[51]:


forecast = client_df[
    ["id", "forecast_cons_12m",
    "forecast_cons_year","forecast_discount_energy","forecast_meter_rent_12m",
    "forecast_price_energy_off_peak","forecast_price_energy_peak",
    "forecast_price_pow_off_peak","churn"
    ]
]


# In[52]:


forecast


# In[55]:


fig, axs = plt.subplots(nrows=7, figsize=(18,50))

# Plot histogram
plot_distribution(client_df, "forecast_cons_12m", axs[0])
plot_distribution(client_df, "forecast_cons_year", axs[1])
plot_distribution(client_df, "forecast_discount_energy", axs[2])
plot_distribution(client_df, "forecast_meter_rent_12m", axs[3])
plot_distribution(client_df, "forecast_price_energy_off_peak", axs[4])
plot_distribution(client_df, "forecast_price_energy_peak", axs[5])
plot_distribution(client_df, "forecast_price_pow_off_peak", axs[6])


# ## Contract Type

# In[56]:


contract_type = client_df[['id', 'has_gas', 'churn']]
contract = contract_type.groupby([contract_type['churn'], contract_type['has_gas']])['id'].count().unstack(level=0)
contract_percentage = (contract.div(contract.sum(axis=1), axis=0) * 100).sort_values(by=[1], ascending=False)


# In[57]:


plot_stacked_bars(contract_percentage, 'Contract type (with gas')


# ## margins

# In[58]:


margin = client_df[['id', 'margin_gross_pow_ele', 'margin_net_pow_ele', 'net_margin']]


# In[60]:


fig, axs = plt.subplots(nrows=3, figsize=(18,20))
# Plot histogram
sns.boxplot(margin["margin_gross_pow_ele"], ax=axs[0])
sns.boxplot(margin["margin_net_pow_ele"],ax=axs[1])
sns.boxplot(margin["net_margin"], ax=axs[2])


#  There are few outliers present here, will deal it in feature engineering

# ## Subscribed Power

# In[62]:


power = client_df[['id', 'pow_max', 'churn']]


# In[63]:


fig, axs = plt.subplots(nrows=1, figsize=(18, 10))
plot_distribution(power, 'pow_max', axs)


# In[64]:


others = client_df[['id', 'nb_prod_act', 'num_years_antig', 'origin_up', 'churn']]
products = others.groupby([others["nb_prod_act"],others["churn"]])["id"].count().unstack(level=1)
products_percentage = (products.div(products.sum(axis=1), axis=0)*100).sort_values(by=[1], ascending=False)


# In[65]:


plot_stacked_bars(products_percentage, "Number of products")


# In[66]:


years_antig = others.groupby([others["num_years_antig"],others["churn"]])["id"].count().unstack(level=1)
years_antig_percentage = (years_antig.div(years_antig.sum(axis=1), axis=0)*100)
plot_stacked_bars(years_antig_percentage, "Number years")


# In[67]:


origin = others.groupby([others["origin_up"],others["churn"]])["id"].count().unstack(level=1)
origin_percentage = (origin.div(origin.sum(axis=1), axis=0)*100)
plot_stacked_bars(origin_percentage, "Origin contract/offer")


# In[ ]:




