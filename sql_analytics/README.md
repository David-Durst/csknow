# Queries
1. Running headshots
    1. all hits where hitgroup is head and velocity (computed using tick before and after) is > 170 u/s (rough metric for not walking or standing still)
    2. problem - have some ridiculous velocities (like 500 u/s or an order of magnitude greater compared to game max of 250 u/s)
   

# Data Quality Issues
1. duplicate hurt events - seems like victim hurt by same attacker multiple times on same frame with same weapon
2. game_id being zero - seems like game_id is 0 at start of some games, meaning many unnecessary ticks for game_id 0