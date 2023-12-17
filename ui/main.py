from tkinter import *  
import sys
sys.path.append('C:\\Users\\Nguyen Quang Huy\\OneDrive\\Desktop\\Tree_Classification\\model')
import model_AI
from model_AI import *

#Define app main
main_app = Tk()
main_app.title("Tree Classification with Image")
main_app.geometry("600x600")

#Define load image button
load_image = Button(main_app, text = "Load Image", command = Tree_Data.load_image)
load_image.pack(pady = 20)

#Define show train image button
load_data = Button(main_app, text = "Load Data", command = Tree_Data.train_image)
load_data.pack(pady = 20)

#Define train csv file box
train_csv = Button(main_app, text = "Create train csv file", command = Tree_Data.make_train_csv)
train_csv.pack(pady = 20)

#Define generate tree image button 
generate_tree_image = Button(main_app, text = "Generate tree image", command = Tree_Data.generate_tree_image)
generate_tree_image.pack(pady = 20)

#Define output listbox
output_listbox = Listbox(main_app, width = 50)
output_listbox.pack(pady = 20)


#Run code 
main_app.mainloop()
