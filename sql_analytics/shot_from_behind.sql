select *
from hurt h
    join ticks t on h.tick_id = t.id
    join player_at_tick pat1 on t.id = pat1.tick_id and h.victim = pat1.player_id
    join player_at_tick pat2 on t.id = pat2.tick_id and h.attacker = pat2.player_id
