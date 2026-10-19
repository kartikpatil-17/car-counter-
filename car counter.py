from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Load YOLO model
model = YOLO(r'D:\kartik\YOLO V8 Murtuza\yolo revision\test\running yolo 1\yolov8n.pt')

# Load video
cap = cv2.VideoCapture(
    r"D:\kartik\YOLO V8 Murtuza\yolo revision\test\videos\cars.mp4"
)

# Load mask
mask = cv2.imread(
    r"D:\kartik\YOLO V8 Murtuza\yolo revision\test\car counter project 1\mask.png"
)

if mask is None:
    print("Error: mask.png not found!")
    exit()

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

tracker = Sort(max_age=20 , min_hits=3 ,iou_threshold=0.3) # if id no 1 is lost how long to wait untill it is recovered 

limits = [400,297,673,297]

totalCount = []

while True:

    success, image = cap.read()

    if not success:
        print("Video Finished")
        break

    imgGraphics = cv2.imread("graphics.png",cv2.IMREAD_UNCHANGED)
    image = cvzone.overlayPNG(image,imgGraphics,(0,0))

    # Resize mask to match frame size
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Apply mask
    imgRegion = cv2.bitwise_and(image, mask_resized)

    # Run YOLO on masked image
    results = model(imgRegion, stream=True)

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            w = x2 - x1
            h = y2 - y1

            conf = float(box.conf[0])

            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car", "bus", "truck", "motorbike"] and conf > 0.3:

                # cvzone.cornerRect(image, (x1, y1, w, h), l=9 , rt=5)

                # cvzone.putTextRect(
                #     image,
                #     f'{currentClass} {conf:.2f}',
                #     (max(0, x1), max(35, y1)),
                #     scale=0.8,
                #     thickness=1,
                #     offset=3
                # )
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(image,(limits[0],limits[1]) , (limits[2],limits[3]) , (0,0,255) , 5)


    for result in resultsTracker:
        x1,y1,x2,y2,id = result
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        print(result)
        cvzone.cornerRect(image, (x1, y1, w, h), l=9 , rt=2 ,colorR=(255,0,255))
        cvzone.putTextRect(
                    image,
                    f'{id}',
                    (max(0, x1), max(35, y1)),
                    scale=2,
                    thickness=3,
                    offset=10
                )
        
        cx,cy = x1+w//2 , y1+h//2
        cv2.circle(image,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 20 < cy < limits[1] + 20:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(image,(limits[0],limits[1]) , (limits[2],limits[3]) , (0,255,0) , 5)

    # cvzone.putTextRect(image,f'Count: {len(totalCount)}',(50,50))
    cv2.putText(image,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)

    cv2.imshow("Original", image)
    # cv2.imshow("Masked Region", imgRegion)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()