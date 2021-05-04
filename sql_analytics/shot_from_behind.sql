select *
from hurt h
    join player_at_tick pat1 on h.tick_id = pat1.tick_id and h.victim = pat1.player_id
    join player_at_tick pat2 on h.tick_id = pat2.tick_id and h.attacker = pat2.player_id