import tkinter as tk
from tkinter import ttk,messagebox
from imageai.Detection import ObjectDetection
from imageai.Detection import VideoObjectDetection
import cv2
import os

mainWindow = tk.Tk()
mainWindow.title('Object Detection')
mainWindow.geometry('600x500')

heading = tk.Label(mainWindow, text='Object Detection',font=("Helvetica 12",20))
heading.grid(row=0,columnspan=2,padx=10,pady=10)

inputLabel = tk.Label(mainWindow, text="Input Path").grid(row=1, column=0, padx=(10,20),pady=(30,20))
outputLabel = tk.Label(mainWindow, text="Output Path").grid(row=2, column=0, padx=(10,10))

input_path_entry = tk.Entry(mainWindow)
output_path_entry = tk.Entry(mainWindow)

input_path_entry.grid(row=1, column=1, padx=(0,10), pady=(30, 20))
output_path_entry.grid(row=2, column=1, padx=(0,10), pady = 20)



##################################################################################


def objectDetect():
    input_path = input_path_entry.get()
    input_path_entry.delete(0, tk.END)
    output_path = output_path_entry.get()
    output_path_entry.delete(0, tk.END)
    
    detector = ObjectDetection()
    
    model_path = "./models/yolo-tiny.h5"
    output_path = output_path+'/newimage.jpg'

    detector.setModelTypeAsTinyYOLOv3()

    detector.setModelPath(model_path)

    detector.loadModel()

    detections = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("--------------------------------")
   
   
################################################################################

def videoDetect():
    input_path = input_path_entry.get()
    input_path_entry.delete(0, tk.END)
    output_path = output_path_entry.get()
    output_path_entry.delete(0, tk.END)
    
    model_path="./models/resnet50_coco_best_v2.0.1.h5"

    detector = VideoObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path)
    detector.loadModel()

    video_path = detector.detectObjectsFromVideo(input_file_path=input_path,
                                output_file_path=os.path.join(output_path, "output_video")
                                , frames_per_second=20, log_progress=True, minimum_percentage_probability=40)
    
        
###########################################################################
def cameraDetect():
    input_path = input_path_entry.get()
    input_path_entry.delete(0, tk.END)
    output_path = output_path_entry.get()
    output_path_entry.delete(0, tk.END)
    
    model_path="./models/resnet50_coco_best_v2.0.1.h5"
    camera = cv2.VideoCapture(0)
    detector = VideoObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(model_path)
    detector.loadModel()
    
    video_path = detector.detectObjectsFromVideo(camera_input=camera,
                                output_file_path=output_path+"camera_detected_video"
                                , frames_per_second=20, log_progress=True, minimum_percentage_probability=40)
    
    
    

###########################################################################    

object_button = tk.Button(mainWindow, text='object Detect',command = lambda : objectDetect())
object_button.grid(row=5,column=0)

video_button = tk.Button(mainWindow, text='Video Detect',command = lambda: videoDetect())
video_button.grid(row=5,column=1)

camera_button = tk.Button(mainWindow, text='Camera Detect',command = lambda: cameraDetect())
camera_button.grid(row=5,column=2)





mainWindow.mainloop()