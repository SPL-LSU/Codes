#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import *
from tkinter import ttk


# In[20]:


#root is toplevel window, contains everything else
#otherwise, pass widgets to parent window (such as root)
root = Tk()
root.title("Test")
content = ttk.Frame(root, padding=" 111 12 12 12")

button = ttk.Button(root)
button2 = ttk.Button(root, text="test", command="buttonpressed")
button.grid()

root.geometry("300x300")
root.mainloop()


# In[3]:





# In[4]:


button2['text']


# In[5]:


button2['text']="test2"
button2['text']


# In[6]:


button.configure()


# In[7]:


#bind: captures any event and executes code


# In[8]:


frame = ttk.Frame(content)
frame['padding'] = (5,10)
frame['borderwidth'] = 2
frame['relief'] = 'sunken'


# In[10]:


label = ttk.Label(content, text = "Full name: ")
resultsContents = StringVar()
label['textvariable'] = resultsContents
resultsContents.set('New value to display')


# In[13]:


content.mainloop()


# In[ ]:




