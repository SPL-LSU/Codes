#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import ttk


# In[10]:


app = tk.Tk()
app.geometry('300x300')

def callback(event):
    print("New Element Selected")

FONT = ("Ariel", 16, "bold")

labelTop = tk.Label(app, text = "Training Data Sets")
labelTop.grid(column=0, row=0)

comboData = ttk.Combobox(app, values=["Training Set 1", "Training Set 2", "Training Set 3"], font=FONT, state="readonly")
comboData.bind("<<ComboboxSelected>>", callback)
app.option_add("*TCombobox*Listbox.font", FONT)

comboData.grid(column=0,row=1)
comboData.current(0)

print(comboData.get())

app.mainloop()


# In[ ]:




