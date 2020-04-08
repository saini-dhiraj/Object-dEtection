from imageai.Detection import VideoObjectDetection
import os
import cv2

execution_path = os.getcwd()
model_path="./models/resnet50_coco_best_v2.0.1.h5"
input_path = "./inputs/tennis_ball.mp4"
output_path = "./output"

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(model_path)
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path=input_path,
                                output_file_path=os.path.join(output_path, "camera_detected_video")
                                , frames_per_second=20, log_progress=True, minimum_percentage_probability=40)