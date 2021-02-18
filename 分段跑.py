#!/usr/bin/env python
# coding: utf-8

# In[305]:


import numpy as np
import pandas as pd


# In[306]:


df = pd.read_csv('../Desktop/vendorChoose.csv',encoding = 'big5') 
print(df) 
pd.DataFrame(df)


# In[307]:


# Create an empty list 
Row_list =[] 
Name_list = []
my_list = []
count_col = df.shape[1] #總共幾行(包括第一行的名稱)
 
    
# Iterate over each row 
for index, rows in df.iterrows(): 
    # Create list for the current row
    
    if my_list == []:
        for j in range(1,count_col,1):
            my_list.append(rows[j])
    else:
        my_list = []
        for j in range(1,count_col,1):
            my_list.append(rows[j])
    #my_list =[rows[1], rows[2], rows[3], rows[4]]

    # append the list to the final list 
    Row_list.append(my_list)
    Name_list.append(rows[0])
    
print('最終: ',Row_list)


# In[308]:


C = None
optimum_choice = None


# In[309]:


a= Row_list
w=[1,1,1,1,1]
I=[1,0,0,1,1]


# In[310]:


# Decision Matrix
a = np.array(a, dtype=np.float).T
print(a)
#assert len(a.shape) == 2, "Decision matrix a must be 2D"


# In[311]:


# Number of alternatives, aspects
(n, J) = a.shape
print((n,J))
print(a.size)
print(n)


# In[312]:


# Weight matrix
w = np.array(w, dtype=np.float)
print(w.size)
assert len(w.shape) == 1, "Weights array must be 1D"
assert w.size == n, "Weights array wrong length, " + "should be of length {}".format(n)


# In[313]:


# Normalise weights to 1
w = w/sum(w)
print(w)


# In[314]:


# Benefit (True) or Cost (False) criteria?
I = np.array(I, dtype=np.int8)
print(I)
assert len(I.shape) == 1, "Criterion array must be 1D"
assert len(I) == n, "Criterion array wrong length, " + "should be of length {}".format(n)


# In[315]:


# Initialise best/worst alternatives lists
ab, aw = np.zeros(n), np.zeros(n)
print(ab, aw)


# In[316]:


#def step1():
""" TOPSIS Step 1
Calculate the normalised decision matrix (self.r)
"""
print(a)
print(' ')
print(np.linalg.norm(a, axis=1)[:, np.newaxis])
print(' ')
r = a/np.array(np.linalg.norm(a, axis=1)[:, np.newaxis])
print(r)


# In[317]:


# def step2(self):
""" TOPSIS Step 2
Calculate the weighted normalised decision matrix
Two transposes required so that indices are multiplied correctly:
"""
print(r.T)
print(' ')
print(w)
print(' ')
v = (w * r.T).T
print(v)


# In[318]:


#def step3(self):
""" TOPSIS Step 3
Determine the ideal and negative-ideal solutions
I[i] defines i as a member of the benefit criteria (True) or the cost
criteria (False)
"""
# Calcualte ideal/negative ideals
ab = np.max(v, axis=1) * I + np.min(v, axis=1) * (1 - I)
aw = np.max(v, axis=1) * (1 - I) + np.min(v, axis=1) * I
print(np.max(v, axis=1))
print('ab=',ab)
print('aw',aw)


# In[319]:


#def step4(self):
""" TOPSIS Step 4
Calculate the separation measures, n-dimensional Euclidean distance
"""
# Create two n long arrays containing Eculidean distances
# Save the ideal and negative-ideal solutions
db = np.linalg.norm(v - ab[:,np.newaxis], axis=0)
dw = np.linalg.norm(v - aw[:,np.newaxis], axis=0)
print('db=',db)
print('dw=',dw)


# In[320]:


# def step5(self):
""" TOPSIS Step 5 & 6
Calculate the relative closeness to the ideal solution, then rank the
preference order
"""
# Ignore division by zero errors
#np.seterr(all='ignore')
# Find relative closeness
C = dw / (dw + db)
print('C = ',C)

showArgsort = C.argsort()
print('showArgsort: ',showArgsort)

get_length = len(C)
#print(length)


#--------好的-----------
#找好的
optimum_choice = C.argsort()[-1]
count_good = 0
opt_idx = []
for i in range(get_length):
    if C[i]==C[optimum_choice]:
        #print('Name: {}(a[{}]) is : {}'.format(Name_list[i],i, a[:, i]))
        count_good+=1
        opt_idx.append(i)

if count_good==0:
    opt_idx.append(optimum_choice)


#--------不好的----------- 
#找不好的
bad_choice = C.argsort()[0]
count_bad=0
bad_idx=[]
for j in range(get_length):
    if C[j]==C[bad_choice]:
        count_bad+=1
        bad_idx.append(j)

if count_bad==0:
    bad_idx.append(bad_choice)

print('--------好的-----------')
print('Good alternative')
for idx in opt_idx:
    print('Name: {}(a[{}]) is : {}'.format(Name_list[idx],idx, a[:, idx]))

print('--------不好的-----------')
print('Bad alternative')
for idx in bad_idx:
    print('Name: {}(a[{}]) is : {}'.format(Name_list[idx],idx, a[:, idx]))


# In[ ]:





# In[107]:


import csv


# In[108]:


# 開啟 CSV 檔案
with open('../Desktop/vendorChoose.csv', newline='') as csvfile:

  # 讀取 CSV 檔案內容
  rows = csv.reader(csvfile)

  # 以迴圈輸出每一列
  for row in rows:
    print(row)


# In[ ]:





# In[ ]:




