import argparse
import pytesseract
import cv2
import numpy as np
import pandas as pd
import pandas.io.sql as sqlio
import os
import re
import psycopg2
from dataclasses import dataclass, asdict

parser = argparse.ArgumentParser()
parser.add_argument("video_file", help="video file to analyze",
                    type=str)
parser.add_argument("output_dir", help="output directory file of visibilities",
                    type=str)
parser.add_argument("spotter", help="the spotter player for the entire video",
                    type=str)
parser.add_argument("player_name_per_color",
                    help="comma separated list of players per color of redPlayer,redGreenPlayer,greenPlayer,greenBluePlayer,bluePlayer,redBluePlayer",
                    type=str)
parser.add_argument("hacking", help="1 if demo is hacking and 0 otherwise",
                    type=int)
parser.add_argument("demo_file", help="name of demo file",
                    type=str)
parser.add_argument("password", help="database password",
                    type=str)
args = parser.parse_args()
print(args)
# get the id for each player
conn = psycopg2.connect(
    host="localhost",
    database="csknow",
    user="postgres",
    password=args.password,
    port=3125)
df_players_and_games = sqlio.read_sql_query(
    'select name, players.id as player_id, g.id as game_id, demo_file from players join games g on players.game_id = g.id',
    conn)

df_deaths = sqlio.read_sql_query(
    f'''
    select t.game_tick_number as game_tick_number
    from kills k
        join players p on k.victim = p.id
        join ticks t on k.tick_id = t.id
        join games g on p.game_id = g.id
    where g.demo_file = {args.demo_file} and
        p.name = '{args.spotter}'
    order by t.game_tick_number;
    '''
)

set_deaths = set(df_deaths['game_tick_number'])

df_round_starts = sqlio.read_sql_query(
    f'''
    select min(t.game_tick_number) as min_game_tick
    from ticks t
    join rounds r on t.round_id = r.id
    join games g on r.game_id = g.id
    where g.demo_file = {args.demo_file}
    group by r.game_id, r.id
    order by r.game_id, r.id;
    '''
)

set_round_starts = set(df_round_starts['min_game_tick'])

def getPlayerId(player_name):
    return df_players_and_games[df_players_and_games['name'] == player_name & 
                                df_players_and_games['demo_file'] == args.demo_file].iloc[0]['player_id']


players = args.player_name_per_color.split(',')


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
seen_cur_tick = []
cur_visibility_events = []
finished_visibility_events = []


@dataclass()
class VisibilityEvent:
    valid: bool
    spotted: str
    start_game_tick: int
    end_game_tick: int
    spotted_id: int
    start_frame_num: int
    end_frame_num: int
    color: str


def visEventToOutputDict(visEvent: VisibilityEvent):
    result = {}
    result['spotter'] = args.spotter
    result['spotted'] = visEvent.spotted
    result['start_game_tick'] = visEvent.start_game_tick
    result['end_game_tick'] = visEvent.end_game_tick
    result['spotter_id'] = getPlayerId(args.spotter)
    result['spotted_id'] = getPlayerId(visEvent.spotted)
    result['demo'] = args.demo_file
    result['hacking'] = args.hacking
    result['start_frame_num'] = visEvent.start_frame_num
    result['end_frame_num'] = visEvent.end_frame_num
    result['color'] = visEvent.color
    return result


for i in range(len(colors)):
    cur_visibility_events.append(VisibilityEvent(False, "", -1, -1, -1, -1, -1, colors[i]))


def resetSeenCurTick():
    for i in range(len(colors)):
        seen_cur_tick[i] = False


def finishTick(player_dead_for_round):
    # using data from last frame, if (didn't see a player or dead) and was in active event for seeing them, finish that event
    for i in range(len(colors)):
        if (player_dead_for_round or not seen_cur_tick[i]) and cur_visibility_events[i].valid:
            # don't need to update end_game_tick and end_frame_num here as that's done every frame that event is valid
            finished_visibility_events.append(cur_visibility_events[i])
            cur_visibility_events[i].valid = False
    resetSeenCurTick()


frame_id = 0
last_frame_tick = -1
player_dead_for_round = False
cap = cv2.VideoCapture(args.video_file)
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # assuming starting on all black frame except for tick number
    if frame_id == 0:
        demoUI_points = cv2.findNonZero(cv2.inRange(frame, np.array([159, 159, 159]), np.array([163, 163, 163])))
        demoUI_rect = cv2.boundingRect(demoUI_points)

    # skip repeated frames
    (_, cap_black_text) = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 175, 255, cv2.THRESH_BINARY_INV)
    cap_black_text_demoUI = cap_black_text[demoUI_rect[1]:demoUI_rect[1] + demoUI_rect[3],
                            demoUI_rect[0]:demoUI_rect[0] + demoUI_rect[2]]
    cv2.imshow('frame', cap_black_text_demoUI)
    text = pytesseract.image_to_string(cap_black_text_demoUI)

    cur_tick_string_match = re.search("Tick:([^\n]+)\/", text)
    if cur_tick_string_match:
        cur_tick = int(cur_tick_string_match.group(1))
    else:
        print("didn't find time on frame " + str(frame_id))
        break

    # if died, then set player_dead_for_round, reset that when next round starts
    if cur_tick in set_deaths:
        player_dead_for_round = True
    elif cur_tick in set_round_starts:
        player_dead_for_round = False

    # demo is over on return to menu when tick becomes 0
    if cur_tick == 0:
        break

    if cur_tick != last_frame_tick:
        finishTick(player_dead_for_round)

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
    # cv2.imshow('frame', cap_rgb)
    for i in range(len(maskCounts)):
        if not player_dead_for_round and maskCounts[i] > 0:
            cur_visibility_events[i].end_game_tick = cur_tick
            cur_visibility_events[i].end_frame_num = frame_id
            if not cur_visibility_events[i].valid:
                cur_visibility_events[i].valid = True
                cur_visibility_events[i].spotted = players[i]
                cur_visibility_events[i].spotted_id = getPlayerId(players[i])
                cur_visibility_events[i].start_game_tick = cur_tick
                cur_visibility_events[i].start_frame_num = frame_id
                cur_visibility_events[i].color = colors[i]

    # cv2.imshow('frame', cap_black_text)
    if frame_id % 1000 == 0:
        print(f'''frame_id {frame_id}''')
        # cv2.imwrite(args.output_dir + "/" + os.path.basename(args.video_file) + "_" + str(frame_id) + ".png", frame)
    frame_id += 1
    last_frame_tick = cur_tick
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

finishTick(True)

df_visibility = pd.DataFrame([visEventToOutputDict(e) for e in finished_visibility_events])
df_visibility.to_csv(args.output_dir + "/" + os.path.basename(args.video_file) + ".csv")

# When everything done, release the capture
cap.release()

cv2.destroyAllWindows()
