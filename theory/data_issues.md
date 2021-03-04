# Data Issues Log
1. getpos makes z ~60 greater than actual position so don't get stuck, then drop
to ground. z coordinates in data are correct
   
1. understadning setang
   1. setang format - pitch (y change in looking) yaw (x change in looking) roll (no difference as always standing straight up)
   2. pitch range (ViewDirectionY) -https://pkg.go.dev/github.com/markus-wa/demoinfocs-golang/v2@v2.5.0/pkg/demoinfocs/common#Player.ViewDirectionY,
   which suggests range is 270->360 (representing -90->0 for looking down) and then 0->90
      1. -90 - looking straight up
      2. -90->0 - looking more and more down
      3. 0 - straight ahead
      4. 0->90 - looking more and more down
      5. 90-180 - straight down
      6. 180.0001-270 - straight up
      7. 270->360 - looking more and more down
      8. 360 - looking straight ahead
      9. 360->450 - looking more and more down
      10. 450 - looking straight down
   3. yaw range (ViewDirectionX)
      1. 0 - looking from B to A
      2. 90 - looking from t spawn to ct spawn
      3. 180 - looking from A to B
      4. 270 - looking from ct spawn to t spawn
      5. 360 - looking from B to A
      5. continues for all periods of 360 degrees
      
   