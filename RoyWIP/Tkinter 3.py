#!/usr/bin/env python
# coding: utf-8

# In[7]:


from tkinter import *
from tkinter import ttk


# In[6]:


root = Tk()


frame = ttk.Frame(root)
#frame['padding']=(5,10)

root.mainloop()


# In[25]:


class Window(Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        self.init_window()
    
    def init_window(self):
    #widgets and shit
        self.pack(fill=BOTH, expand=1)
        frame = ttk.Frame(self, height=1, padding='.5i', width=1)

    
    
    

root = Tk()
root.geometry("300x300")
app = Window(root)
root.mainloop()


# In[27]:


from tkinter import *
master = Tk()
Label(text="one").pack()

separator = Frame(height=2, bd=1, relief=SUNKEN)
separator.pack(fill=X, padx=5, pady=5)

Label(text="two").pack()

mainloop()


# In[30]:


from tkinter import *
from tkinter import ttk

root = Tk()

frame = ttk.Frame(root)
frame['padding']=(100,100)
frame['borderwidth']=2
frame['relief'] = 'sunken'

frame.grid()
root.geometry('300x300')
label = ttk.Label(root, text="Name: ")
root.mainloop()


# In[ ]:




