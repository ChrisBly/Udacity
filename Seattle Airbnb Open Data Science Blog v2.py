#!/usr/bin/env python
# coding: utf-8

# ## Seattle Airbnb Open Data Science Blog

# #### Section 1: Understanding the business

# ## This section is there questions that need to be investigated:
# 
# > 1: Is there a relationship between the day price of the Airbnb place and number of guests?
# 
# > 2: What makes a great Airbnb in Seattle, is the number of people that can stay or it it the maximum number of night you can stay?
# 
# > 3: What are the customers that stay in these Airbnb in Seattle saying about the host and location in there review, let's compare please with listing that have high reviews_per_month with low reviews_per_month to determine keywords that make a great Airbnb stay in Settle
# 

# #### Section 2: Understanding the data
# 
# ##### Gather Data:

# > The kaggle Seattle Airbnb Open website contains the three data source for this project.  The kaggle website requires 
# an user account to download the three Seattle Airbnb Open data sets.  I have an kaggle account,however if you do not 
# have an account the link to main kaggle page is below, just follow the links to create an account, as well as the main kaggle Seattle Airbnb Open website and expected data files.
# 
# - Main kaggle website: https://www.kaggle.com
# 
# - kaggle Seattle Airbnb Open website Link: https://www.kaggle.com/airbnb/seattle
#         
# ##### Expected kaggle Seattle Airbnb Open data sets:
# 
# - calendar
# - listings
# - reviews

# #### Describe data 
# 
# 
# ##### Context
# 
# kaggle Seattle Airbnb Open data set contain listing and descriptions of the Airbnb. Also include is the activities with in the area of the homestays in Seattle, WA.
# 
# 
# #### Content
# 
# kaggle Seattle Airbnb Open data set contains
# 
# - Listings
# - full descriptions
# - average review score
# - Reviews including unique id for each reviewer and detailed comments
# - Calendar, including listing id and the price and availability for that day

# ##### 2:1 Load packages

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from wordcloud import WordCloud


# ##### 2:2 Load Data

# In[35]:


# Load Data
calendar = pd.read_csv("calendar.csv")
calendar.head(1)


# In[36]:


# Load Data
listings = pd.read_csv("listings.csv")
listings.head(1)


# In[37]:


# Load Data
reviews = pd.read_csv("reviews.csv")
reviews.head(1)


# ##### 2:3 Exploring data
# 
# ###### Asses Data:

# ##### Visual Assessment - Verifying quality of data section

# ###### Calendar Data Set

# In[38]:


# Access Data 
calendar.head()


# In[39]:


# Access Data 
calendar.tail()


# ##### Visual Assessment 
# 
# - available: When the value is t there a price and f there an Nan Value.
# - price: has a $ sign, this will be removed.

# #### Listings Data Set

# In[40]:


# Access Data 
listings.head()


# In[41]:


# Access Data 
listings.tail()


# ##### Visual Assessment 
# 
# There are 92 Columns in this data set. 
# 
# Summary:
# 
# - space: Contains a description of the space and missing values. 
# - experiences_offered: Appears to only display the word None. 
# - neighborhood_overview: Contains a description of the neighborhood and missing values
# - review_scores_value: Contains review scorce and missing values. 
# - license contains missing values. 
# - jurisdiction_names Appears to only display the word WASHINGTON.
# - cancellation_policy: Contains three different status: flexible,moderate,strict
# - reviews_per_month: Contain the review per a Month and Nan values. 

# #### Reviews Data Set

# In[42]:


# Access Data 
reviews.head()


# In[43]:


# Access Data 
reviews.tail()


# ##### Visual Assessment
# 
# - listing_id: Listing Airbnb Id
# - id: ID field. Not sure what this relates to. Further investigation. 
# - date: Date of review
# - reviewer_id - ID of reviewer
# - reviewer_name: Name of reviewer
# - comments - Review Comments

# ##### Programmatic assessment - Verifying quality of data section

# #### Calendar Data Set

# #### Checking Data set Structure

# In[44]:


calendar.shape # Access Data 


# There are 1393570 rows and 4 columns in the calendar data set

# In[45]:


calendar.info() # Access Data 


# #### Data Types and Missing Values
# 
# ##### Data Types
# 
# - listing_id  int64 - Needs to string.
# - date        object - Date Format
# - available   object - Format Correct.
# - price       object - Change to Float
# 
# ##### Missing Values
# 
# - price contains 934542 values. 

# #### Basic Statistics

# In[46]:


calendar.describe() # Basic Statistics


# ##### Checking for Missing Values

# In[47]:


calendar.isnull().sum().sort_values(axis = 0, ascending = False) # Checking for Null values


# #### Visualisation Missing Values. 

# ### Create a heatmap Function for Missing Values

# In[48]:


def plot_NA(df):
    df 
    # heatmap for missing values .
    return sns.heatmap(df.isnull(), cbar=False) 


# In[49]:


plot_NA(df = calendar)


# #### Checking for Duplciates

# In[50]:


calendar.duplicated().sum() # checking for duplicates


# #### Checking date range

# In[51]:


calendar['date'].max() # checking for max values


# In[52]:


calendar['date'].min() # checking for Min value


# #### Checking available column and how it relates to price:

# In[53]:


calendar['available'].unique() # checking for unique values


# In[54]:


calendar[calendar['available'].str.contains('t')] # checking for value T in the avaiable column. 


# In[55]:


calendar[calendar['available'].str.contains('f')] # checking for value f in the avaiable column. 


# #### Listings Data Set

# #### Checking Data set Structure

# In[56]:


listings.shape # Access Data


# The listing data set contains 3818 rows and 92 columns 

# In[57]:


listings.info() # Access Data


# #### Basic Statistics

# In[58]:


listings.describe() # Basic Statistics


# In[59]:


listings['price'].min() # checking for min price. 


# In[60]:


listings['price'].max() # checking for max price. 


# ##### Checking for Missing Values

# In[61]:


listings.isnull().sum().sort_values(axis = 0, ascending = False) # checking for missing values


# #### Visualisation Missing Values. 

# In[62]:


plot_NA(df = listings) # use heat map Function for missing values


# #### Review

# #### Checking Data set Structure

# In[63]:


reviews.shape # access data


# The review data set contains 84849 rows and 6 columns.

# In[64]:


reviews.info() # access data


# #### Basic Statistics

# In[66]:


reviews.describe()


# ##### Checking for Missing Values

# In[67]:


reviews.isnull().sum().sort_values(axis = 0, ascending = False) # checking for missing values


# #### Visualisation Missing Values. 

# In[68]:


plot_NA(df = reviews) # use heat map Function for missing values


# #### Checking for Duplciates

# In[69]:


reviews.duplicated().sum() # checking for duplicates


# #### Section 3: Preparation of data

# ##### 3:1 Cleaning Section
# 
# #### Clean Data

# #### 3:2 Report of data cleaning

# #### Issues:
# 
# #### Drop Columns that are not required.
# 
# 
# 
# #### Drop Columns that are not required.
# 
# 
# ##### calendar
# 
# - Available
# 
# ##### Listing
# 
# - listing_url
# - scrape_id
# - last_scraped
# - summary
# - space
# - description
# - experiences_offered
# - neighborhood_overview
# - notes
# - transit
# - thumbnail_url
# - medium_url
# - picture_url
# - host_thumbnail_url
# - host_picture_url
# - street
# - amenities
# - square_feet
# - calendar_updated'
# 
# #### Missing Values:
# 
# ##### calendar
# 
# - Price contains missing values, however this relate to days, where the listing is unavailable. Set to Unavailble.
# 
# ##### calendar
# 
#  
# ##### Remove the Dollar Sign and , from follow data set and columns:
# 
# ###### calendar_clean
# - price
# 
# ##### listings_clean
# 
# - price
# - weekly_price
# - monthly_price
# - security_deposit
# - cleaning_fee
# 
# 
# ##### Define: Change the column format:
# 
# ###### calendar_clean
# 
# - listing_id convert to a string.
# - date convert to date format
# - price convert to a float
# 
# ###### listings_clean
# 
# - id convert to a string.
# - host_id convert to a string.
# - price convert to a float
# - weekly_price convert to a float
# - monthly_price convert to a float
# - security_deposit convert to a float
# - cleaning_fee convert to a float
# 
# 
# ######  reviews_clean
# 
# - listing_id convert to a string.
# - id convert to a string.
# - date convert to date format
# - reviewer_id convert to a string.
# 
# ##### Define: Rename columns as there a dupicate names in the data set. 
# 
# ###### listings_clean
# - id rename to listing_id
# - price rename to listing_price
# 
# ###### calendar_clean
# 
# - date to calendar_date
# - price to day_price
# 
# ###### reviews_clean
# - date to review_date
# - id to reviews_id
# 
# ##### Reset the data set index after merging. 
# 
# #### Define: Split the data set too using review per a month.  
# 
# - Then extract review comments sectionfor word cloud.
# 
# ### Missing Values
# 
# - This project is not using machine learning or Deep learning. 
# 
# ### calendar_clean
# 
# - Missing values are being replace with 0 in the price columns, as this is when the places are not available and does not impact our corr visuals. 
# 
# ### listings_clean
# 
# - Missing values are being replace with 0, as this is when the places are not available and does not impact our corr visuals for the following columns:
# 
# - monthly_price
# - security_deposit
# - weekly_price
# - cleaning_fee
# 
# - Other missing values are being remove from the data set.
# 
# 
# 

# Code

# In[70]:


calendar_clean = calendar.copy() # copying data for cleaning 


# In[71]:


listings_clean = listings.copy()  # copying data for cleaning 


# In[72]:


reviews_clean = reviews.copy()  # copying data for cleaning 


# Test

# In[73]:


calendar_clean.head(1) # testing 


# In[74]:


listings_clean.head(1) # testing 


# In[75]:


reviews_clean.head(1) # testing 


# ###### Define: Remove columns that are not required for this project:
# 
# > ###### calendar_clean Data set:
#     
# - available
# 
# > ###### listings_clean
# 
# - listing_url
# - scrape_id
# - last_scraped
# - summary
# - space
# - description
# - experiences_offered
# - neighborhood_overview
# - notes
# - transit
# - thumbnail_url
# - medium_url
# - picture_url
# - host_thumbnail_url
# - host_picture_url
# - street
# - amenities
# - square_feet
# - calendar_updated'
# 
# 
# 
# > ###### Review
# 
# 

# Code

# In[76]:


calendar_clean.drop(['available'],axis=1,inplace=True) # drop columns


# In[77]:


listings_clean.drop(['listing_url','scrape_id','last_scraped','summary','space','description','experiences_offered','neighborhood_overview','notes','transit','thumbnail_url','medium_url','picture_url','host_thumbnail_url','host_picture_url','street','amenities','square_feet','calendar_updated'],axis=1,inplace=True) # drop columns


# In[78]:


listings_clean.drop(['xl_picture_url','host_url','host_about','requires_license','license','jurisdiction_names','require_guest_profile_picture','require_guest_phone_verification','country_code','country'],axis=1,inplace=True) # drop columns


# Test

# In[79]:


calendar_clean.head(1) # testing 


# In[80]:


listings_clean.head(1) # testing 


# In[81]:


reviews_clean.head(1) # testing 


# ##### Define: Replace Null values in the calendar_clean price column with $0 for plotting time series Graph.

# Code

# In[82]:


calendar_clean['price'] = calendar_clean['price'].fillna('$0') # replace nan 


# Test
#  

# In[83]:


calendar_clean.isnull().sum() # Checking


# ##### Define: Replace Null values in the following Columns With Zero for plotting time series Graph
# 
# - monthly_price
# - security_deposit
# - weekly_price 
# - cleaning_fee 
# 

# Code

# In[84]:


listings_clean['monthly_price'] = listings_clean['monthly_price'].fillna('$0')
listings_clean['security_deposit'] = listings_clean['security_deposit'].fillna('$0')
listings_clean['weekly_price'] = listings_clean['weekly_price'].fillna('$0')
listings_clean['cleaning_fee'] = listings_clean['cleaning_fee'].fillna('$0')


# Test

# In[85]:


listings_clean.isnull().sum().sort_values(axis = 0, ascending = False)


# ##### Define: Removing values that have missng values. 

# Code

# In[86]:


listings_clean.dropna(inplace=True)


# Test

# In[87]:


listings_clean.isnull().sum().sort_values(axis = 0, ascending = False)


# ##### Define: Remove the Dollar Sign and , from follow data set and colums:
# 
# ###### calendar_clean
# - price
# 
# ##### listings_clean
# 
# - price
# - weekly_price
# - monthly_price
# - security_deposit
# - cleaning_fee
# 

# Code

# In[88]:


calendar_clean['price'] = calendar_clean['price'].apply(lambda x: x.replace('$',''))


# In[89]:


calendar_clean['price'] = calendar_clean['price'].apply(lambda x: x.replace(',',''))


# In[90]:


listings_clean['price'] = calendar_clean['price'].apply(lambda x: x.replace('$',''))


# In[91]:


listings_clean['price'] = listings_clean['price'].apply(lambda x: x.replace(',',''))


# In[92]:


listings_clean['weekly_price'] = listings_clean['weekly_price'].apply(lambda x: x.replace('$',''))


# In[93]:


listings_clean['weekly_price'] = listings_clean['weekly_price'].apply(lambda x: x.replace(',',''))


# In[94]:


listings_clean['monthly_price'] = listings_clean['monthly_price'].apply(lambda x: x.replace('$',''))


# In[95]:


listings_clean['monthly_price'] = listings_clean['monthly_price'].apply(lambda x: x.replace(',',''))


# In[96]:


listings_clean['security_deposit'] = listings_clean['security_deposit'].apply(lambda x: x.replace('$',''))


# In[97]:


listings_clean['security_deposit'] = listings_clean['security_deposit'].apply(lambda x: x.replace(',',''))


# In[98]:


listings_clean['cleaning_fee'] = listings_clean['monthly_price'].apply(lambda x: x.replace('$',''))


# In[99]:


listings_clean['cleaning_fee'] = listings_clean['monthly_price'].apply(lambda x: x.replace(',',''))


# Test

# In[100]:


calendar_clean['price'].unique()


# In[101]:


listings_clean['price'].unique()


# In[102]:


listings_clean['weekly_price'].unique()


# In[103]:


listings_clean['monthly_price'].unique()


# In[104]:


listings_clean['cleaning_fee'].unique()


# ##### Define: Change the column format:
# 
# ###### calendar_clean
# 
# - listing_id convert to a string.
# - date convert to date format
# - price convert to a float
# 
# ###### listings_clean
# 
# - id
# - host_id
# - price
# - weekly_price
# - monthly_price
# - security_deposit
# - cleaning_fee
# 
# 
# ######  reviews_clean
# 
# - listing_id convert to a string.
# - id convert to a string.
# - date convert to date format
# - reviewer_id convert to a string.

# Code

# In[105]:


calendar_clean['listing_id'] = calendar_clean['listing_id'].astype(str)
calendar_clean['date'] = pd.to_datetime(calendar_clean['date'])
calendar_clean['price'] = calendar_clean['price'].astype(float)


# In[106]:


listings_clean['id'] = listings_clean['id'].astype(str)
listings_clean['host_id'] = listings_clean['host_id'].astype(str)
listings_clean['price'] = listings_clean['price'].astype(float)
listings_clean['weekly_price'] = listings_clean['weekly_price'].astype(float)
listings_clean['monthly_price'] = listings_clean['monthly_price'].astype(float)
listings_clean['security_deposit'] = listings_clean['security_deposit'].astype(float)
listings_clean['cleaning_fee'] = listings_clean['cleaning_fee'].astype(float)


# In[107]:


reviews_clean['listing_id'] = reviews_clean['listing_id'].astype(str)
reviews_clean['id'] = reviews_clean['id'].astype(str)
reviews_clean['date'] = pd.to_datetime(reviews_clean['date'])
reviews_clean['reviewer_id'] = reviews_clean['reviewer_id'].astype(str)


# Test

# In[108]:


calendar_clean.info()


# In[109]:


listings_clean.info()


# In[110]:


reviews_clean.info()


# ##### Define: Rename columns as there a dupicate names in the data set. 
# 
# ###### listings_clean
# - id rename to listing_id
# - price rename to listing_price
# 
# ###### calendar_clean
# 
# - date to calendar_date
# - price to day_price
# 
# ###### reviews_clean
# - date to review_date
# - id to reviews_id
# 

# Code

# In[111]:


listings_clean.rename({'id':'listing_id'},axis=1,inplace=True)


# In[112]:


listings_clean.rename({'price':'listing_price'},axis=1,inplace=True)


# In[113]:


calendar_clean.rename({'date':'calendar_date'},axis=1,inplace=True)


# In[114]:


calendar_clean.rename({'price':'day_price'},axis=1,inplace=True)


# In[115]:


reviews_clean.rename({'date':'review_date'},axis=1,inplace=True)


# In[116]:


reviews_clean.rename({'id':'reviews_id'},axis=1,inplace=True)


# Test

# In[117]:


listings_clean.info()


# In[118]:


calendar_clean.info()


# In[119]:


reviews_clean.info()


# #### 3.5: Merged data

# ##### Define: Merging data Set:
# 
# ###### Merge 1: listings_clean - calendar_clean
# 
# - Left Join on calendar_clean
# - Merge on Primary key: listing_id
# 
# ###### Merge 2: Merge 1 - reviews_clean
# 
# - Left Join on reviews_clean
# - Merge on Primary key: listing_id

# code

# In[120]:


merge_1 = pd.merge(listings_clean, calendar_clean, how='left', on='listing_id')


# In[121]:


merge_2 = pd.merge(merge_1, reviews_clean, how='left', on='listing_id')


# Test

# In[122]:


merge_1.info()


# In[123]:


merge_2.info()


# In[124]:


merge_2.isnull().sum()


# In[125]:


merge_2.duplicated().sum()


# ### Define: Reset the data set index after merging. 

# Code

# In[126]:


merge_2.reset_index(inplace=True, drop=False)


# #### Define: Split the data set too using review per a month.  
# 
# - #### Then extract review comments section 
# 

# Code

# In[127]:


word_1 = merge_2[merge_2['reviews_per_month'] >= 3]


# In[128]:


word_2 = merge_2[merge_2['reviews_per_month'] < 3]


# In[129]:


wordscloud_1 = word_1['comments']


# In[130]:


wordscloud_2 = word_2['comments']


# Test

# In[131]:


wordscloud_1.shape


# In[132]:


wordscloud_2.shape


# #### Define: Remove duplciate review comments.

# Code

# In[133]:


wordscloud_1.drop_duplicates(inplace=True)
wordscloud_2.drop_duplicates(inplace=True)


# Test

# In[134]:


wordscloud_1.duplicated().sum()


# In[135]:


wordscloud_2.duplicated().sum()


# #### Section 4: Modelling
# 
# ##### Model

# ##### Visualize

# > #### Is there a relationship between the day price of the Airbnb place and number of guests?

# In[136]:


# calculate the correlation matrix
corr = merge_2.corr()


# ### Create a heatmap function

# In[142]:


def heatmap(df):
    df 
    # Generate a mask for the upper triangle
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(600, 30, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    return sns.heatmap(df, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[143]:


heatmap(df=corr) # use heatmap function


# Let's look at the correlation heatmap which basically means: a mutual relationship or connection between two or more things when using a dictionary reference.  In reference to this heat map above , the darker regions of the heatmap indicted a relationship between variables or makers.  These makers are the column names in the table, which as been converted to heatmap.   The colours in the heatmap represent positive or negative relationships between the variables/makers.  When looking at the day_price, we can see a strong relationship between accommodates, bathrooms, bedrooms and beds. 

# > #### What makes a great Airbnb in Seattle, is the number of people that can stay or it it the maximum number of night you can stay?

# #### Create a clustermap Function:

# In[137]:


def clustermap(df):
    df 
    # clustermap for missing values .
    return sns.clustermap(df) 


# In[138]:


clustermap(df=corr) # use clustermap function


# This different type of heatmap, it clusters the variable/columns names together and shows the positive or negative relationships between the variable/columns.  The darker colours  indict a negative relationship, while the lighter colours indict positive relationship.  Looking at this heatmap, we can determine the number of people that can stay at the location  has great impact, than maximum nights.  This is shown by the guests_included, bedrooms, beds and bathrooms, being red hot.  

# > #### What are the customers that stay in these Airbnb in Seattle saying about the host and location in there review, let's compare please with listing that have high reviews_per_month with low reviews_per_month to determine keywords that make a great Airbnb stay in Settle?

# #### High reviews_per_month

# ### Create a word cloud function

# In[139]:


def word_cloud(df):
    df 
    # Create the wordcloud object
    wordcloud = WordCloud().generate(str(df))
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.figure(figsize=(600,800))
    return plt.show()


# In[140]:


word_cloud(df=wordscloud_1) # use the wordcloud function


# Looking at the keywords in the High reviews_per_month word cloud, we can see the Host, comfortable, Apartment, clean are some of the keywords that might a High reviews_per_month in Seattle

# Low reviews_per_month  wordcloud:

# In[141]:


word_cloud(df=wordscloud_2) # use the wordcloud function


# Looking at the keywords in the low reviews_per_month word cloud, we can see the Host,  Great location, clean  are some of the keywords positive words, however there are two keywords that are negative in the word clouds, there related to canceled days and reservation canceled, clearly these have an impact and effective the perception of the  Seattle Airbnb.
# 

# #### Section 5: Evaluation

# ### Conclusion: 
# What makes are great Seattle Airbnb experience,  looking at the result above we can determine that the number of rooms and beds can have an effective on the Seattle Airbnb experience.  We can also determine that the  number of guests_included,  bathrooms are part of the selection process for people looking to stay at the Airbnb location.  
# When looking at the review comments, we can see that the host can add a positive experience to the Airbnb location, however cancellation of the reservation or number of canceled days can impact to listings review. 
# 

# #### Section 6: Setting out

# https://medium.com/@chrisbnsw/what-makes-a-great-airbnb-in-settle-c506ba84c4e5?sk=28bcd932ccb628e47648bdb576918fc9

# In[ ]:




