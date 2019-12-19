#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly.**
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


df = pd.read_csv('ab_data.csv')
df.head()


# b. Use the cell below to find the number of rows in the dataset.

# In[3]:


# number of rows in the dataset using 
df.shape


# The number of rows in 'ab_data.csv'data set is 294478

# c. The number of unique users in the dataset.

# In[4]:


# The number of unique users in the dataset
df['user_id'].nunique()


# the number of unique users in the 'ab_data.csv' dataset is 290584

# d. The proportion of users converted.

# In[5]:


converted_arr = np.array(df['converted'])


# In[6]:


p=converted_arr.mean()
p


# e. The number of times the `new_page` and `treatment` don't match.

# In[7]:


no_match = df.query("(group == 'control' and landing_page == 'new_page') or (group == 'treatment' and landing_page == 'old_page')").shape[0] 
print('Number of times new_page and treatment dont match : ',no_match)


# f. Do any of the rows have missing values?

# In[8]:


# missing values
df.isnull().sum()


# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to figure out how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[9]:


#creat new data set df2 and display the 50 head raws
df2 = df.query("(group == 'treatment' and landing_page == 'new_page') or (group == 'control' and landing_page == 'old_page')")
df2.head(50)


# In[10]:


# the number of rows in new data frame 
df2.shape


# In[11]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[12]:


# the unique user_ids in df2
df2['user_id'].nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[53]:


#print the user_id repeated in df2 by calling duplicated function
duplicated_ids = df2.loc[df2.user_id.duplicated(), 'user_id']
print(duplicated_ids) 


# c. What is the row information for the repeat **user_id**? 

# In[54]:


#print row information for the repeat user_id
df2[df2.user_id.duplicated()]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[15]:


#Remove one of the rows with a duplicate user_id in df2 by using drop function
df2=df2.drop_duplicates(['user_id'])


# `4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[16]:


# probability of an individual converting in df2
df2['converted'].mean()


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[22]:


#print probability was in the `control` group they converted by using
df2[df2['group'] == 'control']['converted'].mean()


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[19]:


#print probability they converted treatment group 
df2[df2['group'] == 'treatment']['converted'].mean()


# d. What is the probability that an individual received the new page?

# In[23]:


size_new = df2.query('landing_page == "new_page"').user_id.size 
size_new/ df2.user_id.size


# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.

# **From above results we can find below information :
# 
# a) probability of conversion : 0.11959708724499628
# 
# b) probability of conversion when individual was in the control group : 0.1203863045004612
# 
# c) probability of conversion when individual was in the treatment group : 0.11880806551510564
# 
# d) probability of individual receiving a new page : 0.5000619442226688.**

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# H0: 𝑝𝑛𝑒𝑤 <= 𝑝𝑜𝑙𝑑 
# 
# H1: 𝑝𝑛𝑒𝑤 >𝑝𝑜𝑙𝑑 

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[24]:


#print the conversion rate for  𝑝𝑛𝑒𝑤  under the null by using mean function
𝑝𝑛𝑒𝑤 = df2['converted'].mean()
print (𝑝𝑛𝑒𝑤)


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[28]:


#print the conversion rate for 𝑝𝑜𝑙𝑑 under the null by using mean function
𝑝𝑜𝑙𝑑=df2['converted'].mean()
𝑝𝑜𝑙𝑑


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[29]:


𝑛𝑛𝑒𝑤 = df2.query('group == "treatment"').converted.count()
𝑛𝑛𝑒𝑤


# d. What is $n_{old}$, the number of individuals in the control group?

# In[30]:


𝑛𝑜𝑙𝑑 = df2.query('group == "control"').converted.count()
𝑛𝑜𝑙𝑑


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[31]:


new_page_converted = np.random.binomial(𝑛𝑛𝑒𝑤 , 𝑝𝑛𝑒𝑤)
new_page_converted


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[33]:


old_page_converted = np.random.binomial(𝑛𝑜𝑙𝑑 , 𝑝𝑜𝑙𝑑)
old_page_converted


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[34]:


𝑝𝑛𝑒𝑤 - 𝑝𝑜𝑙𝑑


# from the hypothies test we found h0 null hyopthies is true becuse 
# H0: 𝑝𝑛𝑒𝑤 <= 𝑝𝑜𝑙𝑑

# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[45]:


p_diffs = []
for _ in range(10000):
    new_page_converted = np.random.binomial(𝑛𝑛𝑒𝑤,𝑝𝑛𝑒𝑤)
    old_page_converted = np.random.binomial(𝑛𝑜𝑙𝑑, 𝑝𝑜𝑙𝑑)
    p_diff = new_page_converted/𝑛𝑛𝑒𝑤 - old_page_converted/𝑛𝑜𝑙𝑑
    p_diffs.append(p_diff)


# In[46]:


p_diffs = np.array(p_diffs)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[47]:


plt.hist(p_diffs)
plt.title('Distribution of p_diffs')
plt.xlabel('p_diff value')
plt.ylabel('Frequency');


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[48]:


p_diff_d = df[df['landing_page'] == 'new_page']['converted'].mean() -  df[df['landing_page'] == 'old_page']['converted'].mean()
p_diff_d


# In[50]:


(p_diffs > -0.00163679).mean()


# In[51]:


p_diff_proportion = (p_diff_d < p_diffs).mean()
p_diff_proportion


# k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# The null hypothesis H0 is true its give the probability of statistics tested In this case the probability new page doesn't have better than the probability of old page.( H0: 𝑝𝑛𝑒𝑤 <= 𝑝𝑜𝑙𝑑)

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[52]:


import statsmodels.api as sm

convert_old = sum(df2.query("landing_page == 'old_page'")['converted'])
convert_new = sum(df2.query("landing_page == 'new_page'")['converted'])
n_old = len(df2.query("landing_page == 'old_page'"))
n_new = len(df2.query("landing_page == 'new_page'"))


# In[57]:


#print information  convert_old ,convert_new,n_old ,n_new 
print ('convert_old is' , convert_old) ,
print ('convert_new is' , convert_new) ,
print ('n_old is' , n_old) ,
print ('n_new is' , n_new) 


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[58]:


# stats proportions_ztest
z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='smaller')
print('z_score is',z_score)
print('p_value is ',p_value)


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# In[59]:


from scipy.stats import norm
# significant of z-score
print(norm.cdf(z_score))

# for our single-sides test, assumed at 95% confidence level, we calculate: 
print(norm.ppf(1-(0.05)))


# in this case we can't reject the null hypothesis H0, we finded  the old page conversions better than new page conversions. because  no significant difference between old page and new page conversions and we finded the critical value of 1.64485362695 more than  the  z-score of 1.31092419842 
# in this case we  can't reject the null hypothesis H0 

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# Logistic Regression

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[63]:


df2['intercept'] = 1
df2[['new_page','old_page']] = pd.get_dummies(df2['landing_page'])
df2['ab_page'] = pd.get_dummies(df2['group'])['treatment']


# In[64]:


df2.head(2)


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# In[68]:


import statsmodels.api as sm
import scipy.stats as stats
log_model = sm.Logit(df2['converted'],df2[['intercept' ,'ab_page']])
results = log_model.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[70]:


#print the summary 
results.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

# p-value is 0.190 .The p-value here suggests that that new page is not statistically significant as 0.19 > 0.05 . In this section it was a two sided test . However here  test for not equal in our hypotheses whereas in Part II it was  equal.
# 
# Hypothesis in Part III 
# 
# $H_{0}$ : $p_{new}$ - $p_{old}$ = 0
# 
# $H_{1}$ : $p_{new}$ - $p_{old}$ != 0
# 
# Hypothesis in Part II 
# 
# $H_{0}$ : $p_{new}$ <= $p_{old}$
# 
# $H_{1}$ : $p_{new}$ > $p_{old}$
# 
# 
# 
# 
# #P-value is 0.190 which means 'ab_page' is not that significant in predicting whether or not the individual converts. H0 in this model is that the 'ab_page' is totally insignificant in predicting the responses and we cannot reject H0.
# 

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# of course we have  many factor  we can ues it to effect individual converts like gender,cultare  and age group  can effect a significant change.we can find new trends using other factors but there may be some disadvantages like even with new factors we maybe miss some other influencing factors which lead to unreliable and contradictory results compared to previous results 

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[71]:


countries_df = pd.read_csv('countries.csv')
df_join = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')
df_join.head()


# In[72]:


df_join['country'].value_counts()


# In[73]:


# Create the necessary dummy variables
df_join[['CA','UK','US']]=pd.get_dummies(df_join['country'])
df_join.head()


# In[74]:


mod = sm.Logit(df_join['converted'], df_join[['intercept', 'CA', 'UK']])
results = mod.fit()


# In[116]:


results.summary()


# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[77]:


# Fit Your Linear Model And Obtain the Results
mod = sm.Logit(df_join['converted'], df_join[['intercept', 'CA', 'UK','ab_page']])
results = mod.fit()


# In[76]:



results.summary()


# <a id='conclusions'></a>
# ## Finishing Up
# 
# 
# 
# 
# ## Directions to Submit
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[83]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])


# In[ ]:




