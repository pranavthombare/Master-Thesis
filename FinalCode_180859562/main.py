import os
from tkinter import *
from tkinter import messagebox
from camera import *
from inference import FaceNetModel
from fr_utils import img_to_encoding
import time
from PIL import Image, ImageTk, ImageFilter

class face_recog:

    def __init__(self,master,FRmodel,inf):
        master.title("Face Recognition")
        master.geometry("600x600")
        self.master = master
        # self.create_database()
        self.isOpenDB = False
        self.open_database()
        self.FRmodel = FRmodel

        ### Title Frame ###
        self.titleFrame = Frame(master)
        self.titleFrame.pack(side=TOP,fill=X)

        self.titleLabel = Label(self.titleFrame,bg="#546E7A",fg="#f8bbd0",text = "Facial Recognition System for Payments",font=(None, 15, "bold"),padx=5,pady=5,width=400,height=2)
        self.titleLabel.pack(side=TOP,fill=X)


        ### Main Frame ###
        self.mainFrame = Frame(master,height=5,width=400)
        self.mainFrame.pack(fill=X)
        self.firstLabel = Label(self.mainFrame,text="Enter your email id",font=(None,10,"italic"),padx=2)
        self.firstLabel.pack(side=LEFT)
        self.entrywidget = Entry(self.mainFrame)
        self.entrywidget.pack(side=LEFT,padx = 2)
        self.entrywidget.insert(0, "enter mail here")

        ### Registered People ###
        self.secondFrame = Frame(master)
        self.secondFrame.pack()

        self.func()
        
        for key,_ in self.database.items():
            #self.secondLabel = Label(self.secondFrame,text = key)
            #self.secondLabel.pack(side=LEFT,padx = 2,pady = 2)
            cwd = os.getcwd()
            dir1 = os.path.join(cwd,"images")
            filepath = str(key)+".png"
            dir2 = os.path.join(dir1,filepath)
            load = Image.open(dir2)
            photo = ImageTk.PhotoImage(load)
            self.photoLabel = Label(self.secondFrame,image=photo,text=key,compound=BOTTOM)
            self.photoLabel.image = photo
            self.photoLabel.pack(side=LEFT,padx=2)

        
        ### Submit Frame ###
        self.submitFrame = Frame(master)
        self.submitFrame.pack(side=LEFT,fill=X)


        self.submitBtn = Button(self.submitFrame,text="Pay",command=self.pay)
        self.submitBtn.pack(side=LEFT,padx=5)        

        self.addNewFaceBtn = Button(self.submitFrame,text="Add User",command = self.add_new_image)
        self.addNewFaceBtn.pack(side=LEFT ,padx = 5)

        self.updateBtn = Button(self.submitFrame,text="Update",command=self.update_database)
        self.updateBtn.pack(side=LEFT,padx =5)

        self.deleteBtn = Button(self.submitFrame,text="Delete",command=self.delete_face)
        self.deleteBtn.pack(side=LEFT,padx=5)

        self.recogBtn = Button(self.submitFrame,text="Recognise",command = self.recog)
        self.recogBtn.pack(side=LEFT,padx=5)

        self.refreshBtn = Button(self.submitFrame,text='Refresh',command=self.refresh)
        self.refreshBtn.pack(side=LEFT,padx=5)

        self.exitBtn = Button(self.submitFrame, text = "Quit" , command = self.quit)
        self.exitBtn.pack(side = LEFT , padx = 5)

    def func(self):
        for key,_ in self.database.items():
            #self.secondLabel = Label(self.secondFrame,text = key)
            #self.secondLabel.pack(side=LEFT,padx = 2,pady = 2)
            cwd = os.getcwd()
            dir1 = os.path.join(cwd,"images")
            filepath = str(key)+".png"
            dir2 = os.path.join(dir1,filepath)
            load = Image.open(dir2)
            photo = ImageTk.PhotoImage(load)
            self.photoLabel = Label(self.secondFrame,image=photo,text=key,compound=BOTTOM)
            self.photoLabel.image = photo
            self.photoLabel.pack(side=LEFT,padx=2)

    def refresh(self):
        self.master.after(10,self.func())
        
    def recog(self):
        if self.isOpenDB == True:
            self.close_database()
        self.email = self.entrywidget.get()

        ob3 = camCapture(root,3,self.email,inf)

    def pay(self):
        if self.isOpenDB == False:
            self.open_database()
        self.email = self.entrywidget.get()

        ob1 = camCapture(root,1,self.email,inf,self.database[self.email])
        score = inf.verify("face.png",self.email,self.database,self.FRmodel)
        print("Score:-{}".format(score))
        if score<=0.7:
            messagebox.showinfo("Verified","Person Verified and payment is successful under  "+str(self.email))
        else:
            messagebox.showinfo("Access Denied","Person is not "+str(self.email))        

    def create_database(self):
        self.database = {}
        import pickle

        handle = open("encoding.pickle","wb")
        pickle.dump(self.database,handle,protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

    def open_database(self):
        import pickle
        handle = open("encoding.pickle","rb")
        self.database = pickle.load(handle)
        self.isOpenDB = True
    
    def update_database(self):
        self.email = self.entrywidget.get()
        self.close_database()
        ob2 = camCapture(root,2,self.email,inf)

        for widget in self.secondFrame.winfo_children():
            widget.destroy()
        self.func()

    def delete_face(self):
        self.email = self.entrywidget.get()

        if self.isOpenDB == False:
            open_database()
        del self.database[self.email]
        os.remove("images/"+str(self.email)+".png")
        for widget in self.secondFrame.winfo_children():
            widget.destroy()
        self.close_database()
        self.func()

    def close_database(self):
        import pickle
        handle = open("encoding.pickle","wb")
        pickle.dump(self.database,handle,protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
        self.isOpenDB = False

    def quit(self):
        if self.isOpenDB == True:
            self.close_database()
        self.master.quit()
        self.master.destroy()

    def add_new_image(self):
        self.email = self.entrywidget.get()
        self.open_database()
        ob1 = camCapture(root,0,self.email,inf)
        for widget in self.secondFrame.winfo_children():
            widget.destroy()
        self.func()
        if self.database.get(self.email,"") == "":
            self.database[self.email] = img_to_encoding("images/"+self.email+".png",self.FRmodel)
            messagebox.showinfo("Info","Face added to the database")





if __name__ == "__main__":
    root = Tk()
    inf = FaceNetModel()
    FRmodel = inf.returnModel()
    obj = face_recog(root,FRmodel,inf)
    root.mainloop()
