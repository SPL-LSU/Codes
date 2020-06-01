#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://pythonprogramming.net/python-3-tkinter-basics-tutorial/


# In[17]:


from tkinter import *
from PIL import Image, ImageTk

class Window(Frame):
    
    def __init__(self, master = None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()
        
    
    def init_window(self):
        self.master.title("Title")
        #pack allows widget to take full amount of space 
        #of root window/frame
        self.pack(fill=BOTH, expand=1)
        
        quitButton = Button(self, text = "bye", command = self.client_exit)
        quitButton.place(x=0, y=0)
        
        #creates a menu
        menu = Menu(self.master)
        self.master.config(menu=menu)
        #add file to menu
        file = Menu(menu)
        #adds command exit to file menu
        file.add_command(label = "Exit", command = self.client_exit)
        menu.add_cascade(label="File", menu=file)
        #add edit to menu
        edit = Menu(menu)
        edit.add_command(label="Undo")
        
        edit.add_command(label="Show Image", command=self.showImg)
        edit.add_command(label="show Text", command = self.showText)
        
        menu.add_cascade(label="Edit", menu=edit)
        
    
    def showImg(self):
        load = Image.open("test.jpg")
        render = ImageTk.PhotoImage(load)
    
        img = Label(self, image=render)
        img.image = render
        img.place(x=0,y=0)
    
    def showText(self):
        text = Label(self, text="Much fashion")
        text.pack()
    
    
    def client_exit(self):
        root.destroy()
root = Tk()


root.geometry("400x300")
app = Window(root)
root.mainloop()


# In[ ]:




