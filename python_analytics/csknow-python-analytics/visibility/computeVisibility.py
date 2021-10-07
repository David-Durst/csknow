import argparse
import math

import pytesseract
import cv2
import numpy as np
import pandas as pd
import pandas.io.sql as sqlio
import os
import re
import psycopg2
from dataclasses import dataclass
import time

parser = argparse.ArgumentParser()
parser.add_argument("video_file", help="video file to analyze",
                    type=str)
parser.add_argument("output_dir", help="output directory for visibility csvs",
                    type=str)
parser.add_argument("log_dir", help="output directory for logs",
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
parser.add_argument("--show_non_ff", help="show images that aren't fast forwarded past", action='store_true')
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
    where g.demo_file = '{args.demo_file}' and
        p.name = '{args.spotter}'
    order by t.game_tick_number;
    ''', conn
)

series_deaths = df_deaths['game_tick_number']

df_round_starts = sqlio.read_sql_query(
    f'''
    select min(t.game_tick_number) as min_game_tick
    from ticks t
    join rounds r on t.round_id = r.id
    join games g on r.game_id = g.id
    where g.demo_file = '{args.demo_file}'
    group by r.game_id, r.id
    order by r.game_id, r.id;
    ''', conn
)

series_round_starts = df_round_starts['min_game_tick']


def getPlayerId(player_name):
    return df_players_and_games[(df_players_and_games['name'] == player_name) &
                                (df_players_and_games['demo_file'] == args.demo_file)].iloc[0]['player_id']


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
        self.hue_min = int(180 * hue_min / 360)
        self.hue_max = int(180 * hue_max / 360)

    def get_lower_bound(self):
        return np.array([self.hue_min, self.saturation_min, self.saturation_min])

    def get_upper_bound(self):
        return np.array([self.hue_max, self.saturation_max, self.saturation_max])

    def get_pixel_count_in_range(self, cap_hsv):
        mask = cv2.inRange(cap_hsv, self.get_lower_bound(), self.get_upper_bound())
        return cv2.countNonZero(mask)

    def get_nonzero_pixels(self, cap_hsv):
        mask = cv2.inRange(cap_hsv, self.get_lower_bound(), self.get_upper_bound())
        return cv2.findNonZero(mask)


# need two reds since color range wraps
redLow = HSVColorBand(8, 8)
redHigh = HSVColorBand(352, 8)
redGreen = HSVColorBand(60, 16)
green = HSVColorBand(120, 16)
greenBlue = HSVColorBand(180, 16)
blue = HSVColorBand(240, 16)
redBlue = HSVColorBand(300, 16)
anyColor = HSVColorBand(180, 180)

colors = ['red', 'redGreen', 'green', 'greenBlue', 'blue', 'redBlue']
seen_cur_tick = []
cur_visibility_events = []
finished_visibility_events = []


@dataclass()
class VisibilityEvent:
    valid: bool
    valid_for_ff: bool # this allows ff checking even when player is dead
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
    cur_visibility_events.append(VisibilityEvent(False, False, "", -1, -1, -1, -1, -1, colors[i]))
    seen_cur_tick.append(False)


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
        if not seen_cur_tick:
            cur_visibility_events[i].valid_for_ff = False
    resetSeenCurTick()


frame_id = -1
max_tick = -1
# this doesn't get updated when fast forwarding through black frames
last_tick = -1
player_dead_for_round = False
cap = cv2.VideoCapture(args.video_file)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
entire_cap_duration_str = time.strftime("%H:%M:%S", time.gmtime(frame_count / fps))
start_time = time.time()
last_frame = None
mask_threshold = 0

def logState(writeFrame=False):
    elapsed_time = time.time() - start_time
    runtime_duration_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    processed_cap_duration_str = time.strftime("%H:%M:%S", time.gmtime(frame_id / fps))
    print(f'''frame_id {frame_id - 1} / {frame_count}, ''' +
          f'''runtime duration from start {runtime_duration_str}, ''' +
          f'''video duration processed {processed_cap_duration_str} / {entire_cap_duration_str}, ''' +
          f'''last non-ff tick {last_tick} / {max_tick}''')
    if writeFrame:
        cv2.imwrite(args.log_dir + "/" + os.path.basename(args.video_file) + ".png", last_frame)


while (cap.isOpened()):
    frame_id += 1
    if (frame_id - 1) % 1000 == 0:
        logState()

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    last_frame = frame

    # assuming starting on all black frame except for tick number
    if frame_id == 0:
        demoUI_points = cv2.findNonZero(cv2.inRange(frame, np.array([159, 159, 159]), np.array([163, 163, 163])))
        demoUI_rect = cv2.boundingRect(demoUI_points)

    # Convert BGR to HSV for hue checks
    cap_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # if all black and not tracking anything, fast forward
    # don't do it on first frame as want to always have a tick to refer to after start
    if frame_id != 0 and anyColor.get_pixel_count_in_range(cap_hsv) == 0 and \
            all([not cve.valid for cve in cur_visibility_events]):
        continue

    # compute color masks
    maskCounts = [redLow.get_pixel_count_in_range(cap_hsv) + redHigh.get_pixel_count_in_range(cap_hsv),
                  redGreen.get_pixel_count_in_range(cap_hsv),
                  green.get_pixel_count_in_range(cap_hsv),
                  greenBlue.get_pixel_count_in_range(cap_hsv),
                  blue.get_pixel_count_in_range(cap_hsv),
                  redBlue.get_pixel_count_in_range(cap_hsv)]

    # if all previously visible colors are still visible and not in last 50 frames, keep going
    # 10 frames is arbitrary safety so don't run off end
    color_change = False
    if max_tick != -1 and last_tick + 10 < max_tick:
        for i in range(len(colors)):
            if (cur_visibility_events[i].valid_for_ff and maskCounts[i] == 0) or \
                    (not cur_visibility_events[i].valid_for_ff and maskCounts[i] > 0):
                color_change = True
                break
        if not color_change:
            continue
            hit_last_frame = False

    # turn gray and invert image so eaiser to find tick text
    cap_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (_, cap_black_text) = cv2.threshold(cap_gray, 175, 255, cv2.THRESH_BINARY_INV)
    # hand measured on a 720p image that 91->108 out of 0->136 for demo ui bounding box
    # was right y values for text, so take just that part of y region in bounding box
    y_bbox_min = math.floor(demoUI_rect[1] + (91.0 / 136) * demoUI_rect[3])
    y_bbox_max = math.ceil(demoUI_rect[1] + (108.0 / 136) * demoUI_rect[3])
    cap_black_text_demoUI = cap_black_text[y_bbox_min:y_bbox_max, demoUI_rect[0]:demoUI_rect[0] + demoUI_rect[2]]
    # cv2.imshow('frame', cap_black_text_demoUI)
    text = pytesseract.image_to_string(cap_black_text_demoUI, config='--psm 6')

    cur_tick_string_match = re.search('Tick[^\d]*(\d+)[^\d]*\/[^\d]*(\d+)', text)
    if cur_tick_string_match:
        cur_tick = int(cur_tick_string_match.group(1))
        if frame_id == 0:
            max_tick = int(cur_tick_string_match.group(2))
    else:
        print("didn't find tick on frame " + str(frame_id))
        break

    # if died sometimes between, then set player_dead_for_round, reset that when next round starts
    # start range at tick after last (or current tick if same tick across frames)
    # range as doing fast forwarding
    if frame_id > 0:
        range_start = min(last_tick + 1, cur_tick)
        if len(series_deaths[series_deaths.between(range_start, cur_tick, inclusive=True)]) > 0:
            player_dead_for_round = True
        # not an elif because you could fast forward over both a death and a round start
        min_cur_tick_and_last_death = cur_tick
        if len(series_round_starts[series_round_starts.between(range_start, cur_tick, inclusive=True)]) > 0:
            player_dead_for_round = False
            min_last_tick_and_last_death = min(min_cur_tick_and_last_death,
                                               series_round_starts[series_round_starts.between(range_start, cur_tick, inclusive=True)].min())


    # demo is over on return to menu when tick becomes 0
    if cur_tick == 0:
        break

    if cur_tick != last_tick:
        finishTick(player_dead_for_round)

    if args.show_non_ff:
        logState(False)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 8035 in short should be recognized, come back to this later
    #if last_tick >= 8037:
    #    logState(True)
    #    x =2

    # Display the resulting frame
    # cv2.imshow('frame', cap_rgb)
    for i in range(len(maskCounts)):
        # skip recognized players that aren't in the demo
        if players[i] == "" and maskCounts[i] > mask_threshold:
            logState(False)
            print(f'''found non-existent player on frame logged in above line with {maskCounts[i]} pixels''')
            continue
        cur_visibility_events[i].valid_for_ff = maskCounts[i] > mask_threshold
        if not player_dead_for_round and maskCounts[i] > mask_threshold:
            cur_visibility_events[i].end_game_tick = cur_tick
            cur_visibility_events[i].end_frame_num = frame_id
            if not cur_visibility_events[i].valid:
                cur_visibility_events[i].valid = True
                cur_visibility_events[i].spotted = players[i]
                cur_visibility_events[i].spotted_id = getPlayerId(players[i])
                cur_visibility_events[i].start_game_tick = min_cur_tick_and_last_death
                cur_visibility_events[i].start_frame_num = frame_id
                cur_visibility_events[i].color = colors[i]

    last_tick = cur_tick
    if last_tick + 10 >= max_tick:
        print("finished all ticks, skipped last 10 for acceptable error bound")
        break

finishTick(True)

df_visibility = pd.DataFrame([visEventToOutputDict(e) for e in finished_visibility_events])
df_visibility.to_csv(args.output_dir + "/" + os.path.basename(args.video_file) + ".csv", index_label='id')

# save last frame for debugging
logState(True)

# When everything done, release the capture
cap.release()

cv2.destroyAllWindows()
