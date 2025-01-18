import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import copy

color_points = {
    "blue": [deque(maxlen=1024)],
    "green": [deque(maxlen=1024)],
    "red": [deque(maxlen=1024)],
    "yellow": [deque(maxlen=1024)],
}

color_indexes = {"blue": 0, "green": 0, "red": 0, "yellow": 0}
 

available_colors = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255), "yellow": (0, 255, 255)}
selected_color = "blue"

canvas_height, canvas_width = 480, 640
canvas = np.full((canvas_height, canvas_width, 3), (200, 200, 200), dtype=np.uint8)

def overlay_button(image, button, position):
    """Overlay a button (non-transparent) onto the image."""
    x, y = position
    h, w = button.shape[:2]
  
    if button.shape[2] == 4: 
        button = cv2.cvtColor(button, cv2.COLOR_BGRA2BGR)


    if y + h > image.shape[0] or x + w > image.shape[1]:
        return  

    image[y:y + h, x:x + w] = button

def redraw_canvas():
   
    global canvas
    canvas[82:, :, :] = (200, 200, 200)  
    for color, points in color_points.items():
        for point_group in points:
            for i in range(1, len(point_group)):
                if point_group[i - 1] and point_group[i]:
                    cv2.line(canvas, point_group[i - 1], point_group[i], available_colors[color], 2)

def draw_buttons(window):
    """Draw UI buttons on the given window."""
    button_positions = [
        ((20, 1), "CLEAR", clear_img),
        ((105, 1), "BLUE", blue_img),
        ((190, 1), "GREEN", green_img),
        ((275, 1), "RED", red_img),
        ((360, 1), "YELLOW", yellow_img),
        ((445, 1), "UNDO", undo_img),
        ((490, 1), "REDO", redo_img),
    ]
    for pos, label, button in button_positions:
        overlay_button(window, button, pos)

clear_img = cv2.imread('button_clear.png')
blue_img = cv2.imread('button_blue.png')
green_img = cv2.imread('button_green.png')
red_img = cv2.imread('button_red.png')
yellow_img = cv2.imread('button_yellow.png')
undo_img = cv2.imread('button_undo.png')
redo_img = cv2.imread('button_redo.png')

clear_img = cv2.resize(clear_img, (80, 80))
blue_img = cv2.resize(blue_img, (80, 80))
green_img = cv2.resize(green_img, (80, 80))
red_img = cv2.resize(red_img, (80, 80))
yellow_img = cv2.resize(yellow_img, (80, 80))
undo_img = cv2.resize(undo_img, (40, 40))
redo_img = cv2.resize(redo_img, (40, 40))

draw_buttons(canvas)

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

video_capture = cv2.VideoCapture(1)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, canvas_width)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, canvas_height)

undo_stack, redo_stack = [], []

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    draw_buttons(frame)

    results = hands_detector.process(frame_rgb)
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.append((int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])))

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

       
        forefinger_tip = landmarks[8]
        thumb_tip = landmarks[4]
        cv2.circle(frame, forefinger_tip, 5, (0, 255, 0), -1)

        if thumb_tip[1] - forefinger_tip[1] < 30: 
            for color in color_points.keys():
                color_points[color].append(deque(maxlen=1024))
                color_indexes[color] += 1

        elif forefinger_tip[1] <= 65: 
            if 20 <= forefinger_tip[0] <= 100:  
                undo_stack.append((copy.deepcopy(color_points), copy.deepcopy(color_indexes)))
                redo_stack.clear()
                canvas[82:, :, :] = (200, 200, 200)
                for color in color_points.keys():
                    color_points[color] = [deque(maxlen=1024)]
                    color_indexes[color] = 0

            elif 105 <= forefinger_tip[0] <= 185:
                selected_color = "blue"
            elif 190 <= forefinger_tip[0] <= 270:
                selected_color = "green"
            elif 275 <= forefinger_tip[0] <= 355:
                selected_color = "red"
            elif 360 <= forefinger_tip[0] <= 440:
                selected_color = "yellow"

            elif 445 <= forefinger_tip[0] <= 485:  
                if undo_stack:
                    redo_stack.append((copy.deepcopy(color_points), copy.deepcopy(color_indexes)))
                    color_points, color_indexes = undo_stack.pop()
                    redraw_canvas()

            elif 490 <= forefinger_tip[0] <= 535:
                if redo_stack:
                    undo_stack.append((copy.deepcopy(color_points), copy.deepcopy(color_indexes)))
                    color_points, color_indexes = redo_stack.pop()
                    redraw_canvas()

        else:  
            if forefinger_tip[1] > 82:  
                undo_stack.append((copy.deepcopy(color_points), color_indexes.copy()))
                redo_stack.clear()
                if selected_color not in color_points:
                    color_points[selected_color] = [deque(maxlen=1024)]
                    color_indexes[selected_color] = 0
                if color_indexes[selected_color] >= len(color_points[selected_color]):
                    color_points[selected_color].append(deque(maxlen=1024))
                color_points[selected_color][color_indexes[selected_color]].appendleft(forefinger_tip)
                redraw_canvas()

    else:
      
        for color in color_points.keys():
            color_points[color].append(deque(maxlen=1024))
        color_indexes = {color: 0 for color in color_points}

   
    for color, points in color_points.items():
        for point_group in points:
            for i in range(1, len(point_group)):
                if point_group[i - 1] and point_group[i]:
                    cv2.line(frame, point_group[i - 1], point_group[i], available_colors[color], 2)
                    cv2.line(canvas, point_group[i - 1], point_group[i], available_colors[color], 2)

    cv2.imshow("Virtual Paint", frame)
    cv2.imshow("Canvas", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

video_capture.release()
cv2.destroyAllWindows()
