from flask import Flask, render_template, Response
import cv2
import numpy as np
from scipy.spatial import KDTree

app = Flask(__name__)

# Add your OpenCV and color processing code here

def get_limits(color):
    color_bgr = color[::-1]
    c = np.uint8([[color_bgr]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = hsvC[0][0][0]
    lowerLimit = (max(hue - 10, 0), 100, 100)
    upperLimit = (min(hue + 10, 179), 255, 255)
    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)
    return lowerLimit, upperLimit

colors = {
    'Red': [0, 0, 255],
    'Green': [0, 255, 0],
    'Blue': [255, 0, 0],
    'Yellow': [0, 255, 255],
    # Add more colors as needed
}

rainbow_colors_hsv = {
    'Red': (0, 10, 100, 255),
    'Orange': (11, 25, 100, 255),
    'Yellow': (26, 35, 100, 255),
    'Green': (36, 85, 100, 255),
    'Blue': (86, 125, 100, 255),
    'Indigo': (126, 145, 100, 255),
    'Violet': (146, 170, 100, 255),
    'Red2': (171, 180, 100, 255),
}

lab_colors = {name: cv2.cvtColor(np.uint8([[value]]), cv2.COLOR_BGR2Lab)[0][0] for name, value in colors.items()}
lab_values = np.array(list(lab_colors.values()))
kdtree = KDTree(lab_values)

def get_color_name(bgr_color):
    clicked_lab = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2Lab)[0][0]
    _, index = kdtree.query(clicked_lab)
    color_name = list(lab_colors.keys())[index]
    return color_name

def analyze_shade(bgr_color):
    hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
    hue, sat, val = hsv_color
    percentages = {}
    total_percentage = 0
    for color_name, (hue_min, hue_max, sat_min, sat_max) in rainbow_colors_hsv.items():
        if hue_min <= hue <= hue_max and sat_min <= sat <= sat_max:
            percentage = ((hue - hue_min) / (hue_max - hue_min + 1)) * 100
            percentages[color_name] = percentage
            total_percentage += percentage
    if total_percentage > 0:
        for color_name in percentages:
            percentages[color_name] = (percentages[color_name] / total_percentage) * 100
    return percentages

def draw_ui(frame, clicked_color, clicked_position, color_box_size=150):
    if clicked_color is not None:
        lowerLimit, upperLimit = get_limits(color=clicked_color)
        box_x = max(clicked_position[0] - 5, 0)
        box_y = max(clicked_position[1] - 5, 0)
        cv2.rectangle(frame, (box_x, box_y), (box_x + 10, box_y + 10), (0, 255, 0), 2)
        cv2.putText(frame, f"BGR: {clicked_color}", (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"HSV: {lowerLimit}-{upperLimit}", (box_x, box_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        bgr_box = np.full((80, color_box_size, 3), clicked_color, dtype=np.uint8)
        cv2.putText(bgr_box, "BGR", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        color_name = get_color_name(clicked_color)
        cv2.putText(bgr_box, color_name, (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        percentages = analyze_shade(clicked_color)
        y_offset = 60
        for color, percentage in percentages.items():
            text = f"{color}: {percentage:.1f}%"
            cv2.putText(bgr_box, text, (5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15
        frame[10:90, 10:10 + color_box_size] = bgr_box
    return frame

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        if 'clicked_color' in globals():
            frame = draw_ui(frame, clicked_color, clicked_position)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)