import pyautogui
import time

i = 0
while True:
    print(f'''mouse position {i}: {pyautogui.position()}''')
    i += 1
    pyautogui.moveTo(1608, 507)
    time.sleep(2)

