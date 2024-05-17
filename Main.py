import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as M
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as C
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import efficientnet.tfkeras as efn
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
import numpy as np
import json
import ctypes


class Main:
    def __init__ (self,window):
        
        self.file_path=""
        self.model = load_model('best_model.hdf5')
        
        with open("ClassToName.json",'r') as file:
            self.data=json.load(file)

        with open("descriptions.json",'r') as file:
            self.descriptions=json.load(file)
            
        self.root = window
        self.root.title("Flower Identifier")

        self.open_button = ttk.Button(self.root, text="Open Image", command=self.open_image)
        self.open_button.grid(row=0, column=0, pady=10)

        self.predict_button = ttk.Button(self.root, text="Predict", command=self.make_prediction)
        self.predict_button.grid(row=0, column=1, pady=10)

        self.image_label = ttk.Label(self.root,anchor='center')
        self.image_label.grid(row=1, column=0)

        self.image_details = ttk.Label(self.root,anchor='center', text="")
        self.image_details.grid(row=1, column=1)
    
    def make_prediction(self):
        predicted_class_index=self.predict(self.file_path)
        text=""
        desc=self.descriptions[str(predicted_class_index)]
        for key,value in desc.items():
            if key!="identifier":
                text=text+"\n"+f"{key} : {value}".title()
        self.image_details.config(text=text)
        
    def open_image(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("JPEG files", "*.jpg;*.jpeg")])
        if self.file_path:
            img = Image.open(self.file_path)
            img.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            
            
    def predict(self,image_path):
        img = keras_image.load_img(image_path, target_size=(512, 512))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions_single = self.model.predict(img_array)
        predicted_class_index = np.argmax(predictions_single)
        return predicted_class_index

            
        


if __name__ == '__main__':
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    root = tk.Tk()
    root.resizable(False,False)
    for i in range(2):
        root.columnconfigure(i, weight=1)
    root.geometry('700x350')
    root.grid_propagate(False)
    obj=Main(root)
    root.mainloop()
