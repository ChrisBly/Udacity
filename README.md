
## Seattle Airbnb Open Data Science Blog

### 1: Data: 
    
- kaggle Seattle Airbnb Open website Link: https://www.kaggle.com/airbnb/seattle

#### 1.1: Data Content

#### kaggle Seattle Airbnb Open data set contains

- Listings
- full descriptions
- average review score
- Reviews including unique id for each reviewer and detailed comments
- Calendar, including listing id and the price and availability for that day

#### Question investigated:

- 1: Is there a relationship between the day price of the Airbnb place and number of guests?

- 2: What makes a great Airbnb in Seattle, is the number of people that can stay or it it the maximum number of night you can stay?

- 3: What are the customers that stay in these Airbnb in Seattle saying about the host and location in there review, let's compare please with listing that have high reviews_per_month with low reviews_per_month to determine keywords that make a great Airbnb stay in Settle

####  Motivation for the project:

- This data was selected for Udacity Project: Create your own science blog.  The Seattle Airbnb Open was one the data set to select.

### 2: Packages Required for this project:

- pandas as pd
- numpy as np
- matplotlib.pyplot as plt
- seaborn as sns
- pandas.plotting import scatter_matrix
- wordcloud import WordCloud

### 3: A summary of the results of the analysis
    
- #### 1: Is there a relationship between the day price of the Airbnb place and number of guests?

> Analysis: When looking at the day_price, we can see a strong relationship between accommodates, bathrooms, bedrooms and beds. 

- #### 2: What makes a great Airbnb in Seattle, is the number of people that can stay or it it the maximum number of night you can stay?

> Analysis:  Looking at this heatmap, we can determine the number of people that can stay at the location  has great impact, than maximum nights.  This is shown by the guests_included, bedrooms, beds and bathrooms, being red hot.  

- #### 3: What are the customers that stay in these Airbnb in Seattle saying about the host and location in there review, let's compare please with listing that have high reviews_per_month with low reviews_per_month to determine keywords that make a great Airbnb stay in Settle

> Analysis: When looking at the review comments, we can see that the host can add a positive experience to the Airbnb location, however cancellation of the reservation or number of canceled days can impact to listings review. 




```python

```
