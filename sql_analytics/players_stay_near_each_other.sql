select *
from player_at_tick pat1
    join ticks t on pat1.tick_id = t.id
    join rounds r on r.id = t.round_id
    join games g on r.game_id = g.id
    join player_at_tick pat2 on pat1.tick_id = pat2.tick_id and pat1.player_id < pat2.player_id
where g.id = 1 and round_id < 10
limit 10;
