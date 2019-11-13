
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     readonly/train.csv - the training set (all tickets issued 2004-2011)
#     readonly/test.csv - the test set (all tickets issued 2012-2016)
#     readonly/addresses.csv & readonly/latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `readonly/train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `readonly/test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32
#        
# ### Hints
# 
# * Make sure your code is working before submitting it to the autograder.
# 
# * Print out your result to see whether there is anything weird (e.g., all probabilities are the same).
# 
# * Generally the total runtime should be less than 10 mins. You should NOT use Neural Network related classifiers (e.g., MLPClassifier) in this question. 
# 
# * Try to avoid global variables. If you have other functions besides blight_model, you should move those functions inside the scope of blight_model.
# 
# * Refer to the pinned threads in Week 4's discussion forum when there is something you could not figure it out.

# In[79]:

import pandas as pd
import numpy as np

def blight_model():
    
    # Your code here
    train_data=pd.read_csv('readonly/train.csv',encoding = "ISO-8859-1")
    test_data=pd.read_csv('readonly/test.csv',encoding = "ISO-8859-1")
    #dropping the non-eligible columns in train_data-imp
    train_data1=train_data.drop(['payment_amount', 'payment_date','payment_status','balance_due','collection_status','compliance_detail'], axis=1)
    train_data2=train_data1.drop(['violator_name','violation_street_name','mailing_address_str_name','violation_description'], axis=1)
    train_data2=train_data2.drop(['ticket_issued_date','hearing_date'], axis=1)
    train_data2=train_data2.drop(['violation_street_number','mailing_address_str_number'], axis=1)
    train_data2=train_data2.drop(['admin_fee', 'state_fee', 'late_fee', 'discount_amount', 'clean_up_cost', 'judgment_amount',], axis=1)
    train_data2=train_data2.drop(['ticket_id'], axis=1)
    train_data2=train_data2.drop(['non_us_str_code'], axis=1)
    train_data2=train_data2.drop(['violation_zip_code'], axis=1)
    train_data2=train_data2.drop(['grafitti_status'], axis=1)
    test_data2=test_data[['ticket_id','agency_name','inspector_name','city','state','zip_code','country','violation_code','disposition','fine_amount']]
    #converting text labels to numbers
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    train_data2['agency_name'] = labelencoder.fit_transform(train_data2['agency_name'])
    test_data2['agency_name'] = labelencoder.fit_transform(test_data2['agency_name'])
    labelencoder1 = LabelEncoder()
    train_data2['inspector_name'] = labelencoder1.fit_transform(train_data2['inspector_name'])
    test_data2['inspector_name'] = labelencoder1.fit_transform(test_data2['inspector_name'])
    labelencoder2 = LabelEncoder()
    train_data2['city'] = labelencoder2.fit_transform(train_data2['city'])
    test_data2['city'] = labelencoder2.fit_transform(test_data2['city'].astype(str))

    #train_data2['state'] = labelencoder.fit_transform(train_data2['state'])
    #train_data2['zip_code'] = labelencoder.fit_transform(train_data2['zip_code'])
    labelencoder3 = LabelEncoder()
    train_data2['country'] = labelencoder3.fit_transform(train_data2['country'])
    test_data2['country'] = labelencoder3.fit_transform(test_data2['country'])
    labelencoder4 = LabelEncoder()
    train_data2['violation_code'] = labelencoder4.fit_transform(train_data2['violation_code'])
    test_data2['violation_code'] = labelencoder4.fit_transform(test_data2['violation_code'])
    labelencoder5 = LabelEncoder()
    train_data2[ 'disposition'] = labelencoder5.fit_transform(train_data2[ 'disposition'])
    test_data2[ 'disposition'] = labelencoder5.fit_transform(test_data2[ 'disposition'])
    #train_data2[ 'grafitti_status'] = labelencoder.fit_transform(train_data2[ 'grafitti_status'])
    #have to get rid of the NaN and missing values
    train_data3=train_data2.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    test_data3=test_data2.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    test_data4=test_data3
    train_data4=train_data3[~train_data3.state.str.contains("XYZ")]
    labelencoder7 = LabelEncoder()
    train_data4['state'] = labelencoder7.fit_transform(train_data4['state'])
    test_data4['state'] = labelencoder7.fit_transform(test_data4['state'])
    # keeping the rows where zip_code values are of 5 digit
    train_data5 = train_data4[train_data4.astype(str).zip_code.map(len) == 5]
    test_data5 = test_data4[test_data4.astype(str).zip_code.map(len) == 5]
    labelencoder6 = LabelEncoder()
    train_data5['zip_code'] = labelencoder6.fit_transform(train_data5['zip_code'].astype(str))
    test_data5['zip_code'] = labelencoder6.fit_transform(test_data5['zip_code'].astype(str))
    X_test=test_data5.drop('ticket_id',axis=1)
    ticket_id=test_data5['ticket_id']
    X=train_data5.drop('compliance',axis=1)
    y=train_data5[['compliance']]
    #creating random forest model
    # Load libraries
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import datasets
    import numpy as np
    import matplotlib.pyplot as plt
    # Create decision tree classifer object
    clf = RandomForestClassifier(random_state=0, n_jobs=-1)

    # Train model
    model = clf.fit(X, y)
    y_test=model.predict_proba(X_test)[:,1]
    op = pd.Series(y_test)
    op.index=[ticket_id]
    return op # Your answer here


# In[121]:

#blight_model()


# In[80]:

train_data=pd.read_csv('readonly/train.csv',encoding = "ISO-8859-1")


# In[82]:

test_data=pd.read_csv('readonly/test.csv',encoding = "ISO-8859-1")


# In[83]:

#len(train_data.inspector_name.unique())


# In[84]:

#dropping the non-eligible columns in train_data-imp
train_data1=train_data.drop(['payment_amount', 'payment_date','payment_status','balance_due','collection_status','compliance_detail'], axis=1)
train_data2=train_data1.drop(['violator_name','violation_street_name','mailing_address_str_name','violation_description'], axis=1)
train_data2=train_data2.drop(['ticket_issued_date','hearing_date'], axis=1)
train_data2=train_data2.drop(['violation_street_number','mailing_address_str_number'], axis=1)
train_data2=train_data2.drop(['admin_fee', 'state_fee', 'late_fee', 'discount_amount', 'clean_up_cost', 'judgment_amount',], axis=1)
train_data2=train_data2.drop(['ticket_id'], axis=1)
train_data2=train_data2.drop(['non_us_str_code'], axis=1)


# In[85]:

train_data2=train_data2.drop(['violation_zip_code'], axis=1)


# In[86]:

train_data2=train_data2.drop(['grafitti_status'], axis=1)


# In[87]:

#counting the unique values of the column 'ticketid' of train_dat1
#train_data1['ticket_id'].nunique()


# In[88]:

#counting the unique values of the column 'id' of train_dat1
#test_data['ticket_id'].nunique()


# In[89]:

#list(train_data2.columns)


# In[90]:

#train_data2['violation_zip_code', 'city'].apply(LabelEncoder().fit_transform)


# In[91]:

#train_data2['violation_zip_code'] = labelencoder.fit_transform(train_data2['violation_zip_code'])


# In[92]:

test_data2=test_data[['ticket_id','agency_name','inspector_name','city','state','zip_code','country','violation_code','disposition','fine_amount']]


# In[93]:

#converting text labels to numbers
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
train_data2['agency_name'] = labelencoder.fit_transform(train_data2['agency_name'])
test_data2['agency_name'] = labelencoder.fit_transform(test_data2['agency_name'])
labelencoder1 = LabelEncoder()
train_data2['inspector_name'] = labelencoder1.fit_transform(train_data2['inspector_name'])
test_data2['inspector_name'] = labelencoder1.fit_transform(test_data2['inspector_name'])
labelencoder2 = LabelEncoder()
train_data2['city'] = labelencoder2.fit_transform(train_data2['city'])
test_data2['city'] = labelencoder2.fit_transform(test_data2['city'].astype(str))

#train_data2['state'] = labelencoder.fit_transform(train_data2['state'])
#train_data2['zip_code'] = labelencoder.fit_transform(train_data2['zip_code'])
labelencoder3 = LabelEncoder()
train_data2['country'] = labelencoder3.fit_transform(train_data2['country'])
test_data2['country'] = labelencoder3.fit_transform(test_data2['country'])
labelencoder4 = LabelEncoder()
train_data2['violation_code'] = labelencoder4.fit_transform(train_data2['violation_code'])
test_data2['violation_code'] = labelencoder4.fit_transform(test_data2['violation_code'])
labelencoder5 = LabelEncoder()
train_data2[ 'disposition'] = labelencoder5.fit_transform(train_data2[ 'disposition'])
test_data2[ 'disposition'] = labelencoder5.fit_transform(test_data2[ 'disposition'])
#train_data2[ 'grafitti_status'] = labelencoder.fit_transform(train_data2[ 'grafitti_status'])


# In[94]:

#train_data2['state'] = labelencoder.fit_transform(train_data2['state'])


# In[95]:

#finding NULL in the state column
#(pd.isnull(train_data3['state'])==True).sum()


# In[96]:

#train_data2.head()


# In[97]:

#have to get rid of the NaN and missing values
train_data3=train_data2.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
test_data3=test_data2.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)


# In[98]:

test_data4=test_data3


# In[99]:

train_data4=train_data3[~train_data3.state.str.contains("XYZ")]


# In[100]:

labelencoder7 = LabelEncoder()
train_data4['state'] = labelencoder7.fit_transform(train_data4['state'])
test_data4['state'] = labelencoder7.fit_transform(test_data4['state'])


# In[101]:

#test_data4.head()


# In[102]:

# keeping the rows where zip_code values are of 5 digit
train_data5 = train_data4[train_data4.astype(str).zip_code.map(len) == 5]
test_data5 = test_data4[test_data4.astype(str).zip_code.map(len) == 5]


# In[103]:

labelencoder6 = LabelEncoder()
train_data5['zip_code'] = labelencoder6.fit_transform(train_data5['zip_code'].astype(str))
test_data5['zip_code'] = labelencoder6.fit_transform(test_data5['zip_code'].astype(str))


# In[104]:

X_test=test_data5.drop('ticket_id',axis=1)


# In[105]:

ticket_id=test_data5['ticket_id']


# In[106]:

#train_data5['agency_name'].unique()


# In[107]:

#calculating the length of the zip_code 
#len(str(train_data4['zip_code'][0]))


# In[108]:

X=train_data5.drop('compliance',axis=1)
y=train_data5[['compliance']]


# In[109]:

#creating random forest model
# Load libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
# Create decision tree classifer object
clf = RandomForestClassifier(random_state=0, n_jobs=-1)

# Train model
model = clf.fit(X, y)


# In[110]:

y_test=model.predict_proba(X_test)[:,1]


# In[111]:

#y_test.shape


# In[112]:

#t_i=ticket_id.values.tolist()


# In[113]:

op = pd.Series(y_test)


# In[114]:

#op.index=[t_i]
op.index=[ticket_id]


# In[115]:

#op


# In[116]:

#train_data.iloc[1]


# In[117]:

#model.predict_proba(X.iloc[1])[:,1]


# In[118]:

#blight_model()


# In[119]:

#have to use predict_proba() to find probablity of prediction

