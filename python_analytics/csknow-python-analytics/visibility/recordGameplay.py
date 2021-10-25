import pyautogui
import pydirectinput
import time

if False:
    i = 0
    while True:
        print(f'''mouse position {i}: {pyautogui.position()}''')
        i += 1
        time.sleep(2)

pyautogui.moveTo(164, 1185)
pyautogui.click()
time.sleep(2)
with pyautogui.hold('alt'):
    pyautogui.press('f')
pyautogui.press('enter')
pyautogui.press('enter')

time.sleep(25)

pyautogui.moveTo(950, 763)
pyautogui.click()
pyautogui.write('exec gotv_4_preload\n')

time.sleep(25)

pydirectinput.press('`')
pyautogui.write('exec create_bfs_stream\n')

time.sleep(0.5)

pydirectinput.moveTo(620, 639)
pydirectinput.mouseDown(button='left')
pydirectinput.moveTo(405, 326)
pydirectinput.mouseUp(button='left')

pydirectinput.moveTo(950, 763)
pyautogui.click()
pyautogui.write('mirv_streams previewEnd\n')
time.sleep(0.5)
pyautogui.write('demo_resume\n')
pydirectinput.press('`')



