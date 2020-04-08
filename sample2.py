from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()
model_path="./models/resnet50_coco_best_v2.0.1.h5"

camera = cv2.VideoCapture(0)

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(model_path)
detector.loadModel()

video_path = detector.detectObjectsFromVideo(camera_input=camera,
                                output_file_path=os.path.join(execution_path, "camera_detected_video")
                                , frames_per_second=20, log_progress=True, minimum_percentage_probability=40)