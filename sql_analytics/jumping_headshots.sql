select *
from ticks t
         join hurt h on t.id = h.tick_id
         join player_at_tick pat on h.tick_id = pat.tick_id and attacker = player_id
where is_airborne
limit 10;