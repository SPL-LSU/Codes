#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tkinter import *
from tkinter import ttk #themed widget module
#Tutorial Used:

#https://tkdocs.com/tutorial/intro.html


# In[ ]:





# In[2]:


root = Tk()
root.title("Feet to Meters")


def calculate(*args):
    try:
        value = float(feet.get())
        meters.set((.3048*value*10000.+.5)/10000.)
    except ValueError:
        pass


mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column= 0, row = 0, sticky= (N, W, E, S))
root.columnconfigure(0, weight=1) #expand to take up extra space if resized
root.rowconfigure(0, weight = 1)#expand to take up extra space if resized

feet = StringVar()
meters = StringVar()

feet_entry = ttk.Entry(mainframe, width = 7, textvariable = feet)
feet_entry.grid(column = 2, row = 1, sticky=(W,E))

ttk.Label(mainframe, textvariable = meters).grid(column = 2, row = 2, sticky = (W,E))
ttk.Button(mainframe, text = "Calculate", command = calculate).grid(column = 3, row = 3, sticky = W)

ttk.Label(mainframe, text = "feet").grid( column = 3, row = 1, sticky = W)
ttk.Label(mainframe, text = "is equivalent to").grid(column = 1, row = 2, sticky = E)
ttk.Label(mainframe, text = "meters").grid(column = 3, row = 2, sticky = W)

for child in mainframe.winfo_children(): child.grid_configure(padx = 5, pady = 5)
    
feet_entry.focus()
root.bind('<Return>', calculate)

root.mainloop()


# In[3]:


#Widgets: objects that appear on screen, AKA "windows"
root=Tk()
#"root" is toplevel window that contains every other widget
content = ttk.Frame(root)
#other widgets must have
button = ttk.Button(content)


# In[4]:


#lists all the options for your widget
button.configure()
content.configure()


# In[5]:


button = ttk.Button(root, text = 'Test', command = 'buttonpressed')
button.grid()
button['text']
button['text']="test"
button['text']
button.configure('text')


# In[6]:


#Geometry Manager takes widgets and displays them on screen


# In[7]:


#Event Handling: determines what widget an event applies, dispatches action


# In[2]:


from tkinter import *
from tkinter import ttk
root = Tk()
l =ttk.Label(root, text="Starting...")
l.grid()
l.bind('<Enter>', lambda e: l.configure(text='Moved mouse inside'))
l.bind('<Leave>', lambda e: l.configure(text='Moved mouse outside'))
l.bind('<1>', lambda e: l.configure(text='Clicked left mouse button'))
l.bind('<Double-1>', lambda e: l.configure(text='Double clicked'))
l.bind('<B3-Motion>', lambda e: l.configure(text='right button drag to %d,%d' % (e.x, e.y)))
root.mainloop()


# In[8]:


from tkinter import *
from tkinter import ttk
root = Tk()

frame = ttk.Frame(root, borderwidth=10)
frame['borderwidth']=300
label=ttk.Label(frame)
label['text']="Is it working?"
label.grid()
label.configure('text')
button=ttk.Button(frame, text = 'help')
button.grid(column=0, row=0)
check = ttk.Checkbutton(root)
username=StringVar()
name = ttk.Entry(root, textvariable=username)
name.grid(column=0, row=0)
frame.pack
root.mainloop()


# In[27]:


label.configure()


# In[21]:


l


# In[ ]:




