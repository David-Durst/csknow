import argparse
import pytesseract
import cv2
import numpy as np
import pandas as pd
import os
import re
import psycopg2

parser = argparse.ArgumentParser()
parser.add_argument("video_file", help="video file to analyze",
                    type=str)
parser.add_argument("output_dir", help="output directory file of visibilities",
                    type=str)
parser.add_argument("spotter", help="the spotter player for the entire video",
                    type=str)
parser.add_argument("player_name_per_color", help="comma separated list of players per color of redPlayer,redGreenPlayer,greenPlayer,greenBluePlayer,bluePlayer,redBluePlayer",
                    type=str)
parser.add_argument("hacking", help="1 if demo is hacking and 0 otherwise",
                    type=int)
parser.add_argument("demo_file", help="name of demo file",
                    type=int)
parser.add_argument("password", help="database password",
                    type=str)
parser.add_argument("query_file", help="file containing query",
                    type=str)
args = parser.parse_args()

conn = psycopg2.connect(
    host="localhost",
    database="csknow",
    user="postgres",
    password=args.password,
    port=3125)
cur = conn.cursor()

players = args.player_name_per_color.split(',')

cap = cv2.VideoCapture(args.video_file)

# define colors in hsv, allow any value above like 30 as blend with black backgorund ok, need saturation of at least 60
# so filter out smokes and flash
class HSVColorBand:
    hue_min: int
    hue_max: int
    saturation_min: int = 60
    saturation_max: int = 255
    value_min: int = 30
    value_max: int = 255

    def __init__(self, hue_mid, hue_range):
        # Normal H,S,V: (0-360,0-100%,0-100%)
        # OpenCV H,S,V: (0-180,0-255 ,0-255)
        hue_min = hue_mid - hue_range
        hue_max = hue_mid + hue_range
        self.hue_min = 180 * hue_min / 360
        self.hue_max = 180 * hue_max / 360

    def get_lower_bound(self):
        return np.array([self.hue_min, self.saturation_min, self.saturation_max])

    def get_upper_bound(self):
        return np.array([self.hue_max, self.saturation_max, self.saturation_max])

    def get_pixel_count_in_range(self, cap_hsv):
        mask = cv2.inRange(cap_hsv, self.get_lower_bound(), self.get_upper_bound())
        return cv2.countNonZero(mask)

# need two reds since color range wraps
redLow = HSVColorBand(8, 8)
redHigh = HSVColorBand(352, 8)
redGreen = HSVColorBand(60, 16)
green = HSVColorBand(120, 16)
greenBlue = HSVColorBand(180, 16)
blue = HSVColorBand(240, 16)
redBlue = HSVColorBand(300, 16)

colors = ['red', 'redGreen', 'green', 'greenBlue', 'blue', 'redBlue']

spotter = []
spotted = []
start_game_tick = []
end_game_tick = []
spotter_id = []
spotted_id = []
demo = []
hacking = []
start_frame_num = []
end_frame_num = []
color = []

frame_id = 0
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # assuming starting on all black frame except for tick number
    if frame_id == 0:
        demoUI_points = cv2.findNonZero(cv2.inRange(frame, np.array([159, 159, 159]), np.array([163, 163, 163])))
        demoUI_rect = cv2.boundingRect(demoUI_points)

    # skip repeated frames
    (_, cap_black_text) = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 175, 255, cv2.THRESH_BINARY_INV)
    cap_black_text_demoUI = cap_black_text[demoUI_rect[1]:demoUI_rect[1]+demoUI_rect[3],
                                           demoUI_rect[0]:demoUI_rect[0]+demoUI_rect[2]]
    cv2.imshow('frame', cap_black_text_demoUI)
    text = pytesseract.image_to_string(cap_black_text_demoUI)

    cur_tick_string_match = re.search("Tick:([^\n]+)\/", text)
    if cur_tick_string_match:
        cur_tick = int(cur_tick_string_match.group(1))
    else:
        print("didn't find time on frame " + str(frame_id))
        quit(1)


    # Convert BGR to HSV
    cap_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # compute masks
    maskCounts = [redLow.get_pixel_count_in_range(cap_hsv) + redHigh.get_pixel_count_in_range(cap_hsv),
                  redGreen.get_pixel_count_in_range(cap_hsv),
                  green.get_pixel_count_in_range(cap_hsv),
                  greenBlue.get_pixel_count_in_range(cap_hsv),
                  blue.get_pixel_count_in_range(cap_hsv),
                  redBlue.get_pixel_count_in_range(cap_hsv)]

    # Display the resulting frame
    #cv2.imshow('frame', cap_rgb)
    for i in range(len(maskCounts)):
        spotter.append(args.spotter)
        spotted.append(players[i])
        frame_num.append(frame_id)
        game_tick.append(-1)
        color.append(colors[i])

    #cv2.imshow('frame', cap_black_text)
    if frame_id % 1000 == 0:
        print(f'''frame_id {frame_id}''')
        cv2.imwrite(args.output_dir + "/" + os.path.basename(args.video_file) + "_" + str(frame_id) + ".png", frame)
    frame_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

dict_visibility = {'spotter': spotter, 'spotted': spotted, 'frame_num': frame_num, 'game_tick': game_tick, 'color': color}
df_visibility = pd.DataFrame(dict)
df_visibility.to_csv(args.output_dir + "/" + os.path.basename(args.video_file) + ".csv")

# When everything done, release the capture
cap.release()

cv2.destroyAllWindows()