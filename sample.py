from imageai.Detection import ObjectDetection

detector = ObjectDetection()

model_path = "./models/yolo-tiny.h5"
input_path = "./inputs/image2.jpg"
output_path = "./output/newimage.jpg"

detector.setModelTypeAsTinyYOLOv3()

detector.setModelPath(model_path)

detector.loadModel()

detections = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")