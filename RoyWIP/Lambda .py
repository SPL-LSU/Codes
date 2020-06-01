#!/usr/bin/env python
# coding: utf-8

# In[2]:


#the following are equivalent
def identity(x):
    return x
lambda x: x


# In[7]:


identity(1)


# In[8]:


(lambda x: x)(1)


# In[13]:


(lambda x: x+1)(2)


# In[14]:


def add_one(x):
    return x+1
add_one(2)


# In[17]:


#multi arguement lambda:

full_name = lambda first, last: f'Fullname: {first.title()} {last.title()}'


# In[18]:


full_name('first', 'last')


# In[20]:


#or all in one line
(lambda first, last: f'Fullname: {first.title()} {last.title()}')('first', 'last')


# In[29]:


#anonymous functions == lambda functions (essentially)
#anonymous function is one without a name
#in python use lambdas to do this
lambda x,y : x+y


# In[30]:


_(1,2)
#have to invoke following immediately
#underscore calls the last function


# In[34]:


#high order functions
#takes a function, lambda or normal, a argument
high_ord_func = lambda x, func: x +func(x)


# In[35]:


#x +x^2
high_ord_func(2, lambda x: x*x)


# In[ ]:


#lambda's are only useful for lazy people

