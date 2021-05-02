select *
from ticks t
         join hurt h on t.id = h.tick_id
         join player_at_tick pat on h.tick_id = pat.tick_id and attacker = player_id
where remaining_flash_time > 0.0 and hit_group = 1
limit 10;
