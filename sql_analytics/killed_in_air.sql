select *
from ticks t
         join kills k on t.id = k.tick_id
         join player_at_tick pat on k.tick_id = pat.tick_id and victim = player_id
where is_airborne
limit 10;
