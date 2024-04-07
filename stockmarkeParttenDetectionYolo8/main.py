import numpy as np
import win32gui
from PIL import ImageGrab
from ultralyticsplus import YOLO, render_result
import cv2

# load model
model = YOLO('foduucom/stockmarket-pattern-detection-yolov8')

# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image


def capture_dynamic():
    toplist, winlist = [], []

    def enum_cb(hwnd, results):
        winlist.append((hwnd, win32gui.GetWindowText(hwnd)))

    win32gui.EnumWindows(enum_cb, toplist)

    wnd = [(hwnd, title) for hwnd, title in winlist if 'iq option' in title.lower()]

    if wnd:
        wnd = wnd[0]
        hwnd = wnd[0]

        bbox = win32gui.GetWindowRect(hwnd)
        img = ImageGrab.grab((20, 200, 1000, 1000))
        return img
    else:
        return None

# initialize video capture
# Open the video file

# Loop through the video frames
while True:
    # Read a frame from the video
    frame = capture_dynamic()

    if frame == None:
        print("No Window Found! Please Try Again")
        break

    width, height = frame.size

    # Setting the points for cropped image
    left = 0
    top = 400
    right = width
    bottom = height

    # Cropped image of above dimension
    # (It will not change original image)
    # frame = frame.crop((left, top, right, bottom))

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    # cv2.imshow("YOLOv8 Inference", cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

    screen_grab = np.array(annotated_frame)

    image = cv2.resize(cv2.cvtColor(screen_grab, cv2.COLOR_BGR2RGB), (600, 500))
    cv2.imshow("YOLO Reference", image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

