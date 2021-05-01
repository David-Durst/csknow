select pat.tick_id                                      as tick_id,
       pat.player_id,
       pat.view_x,
       g.game_tick_rate,
       t.demo_tick_number,
       t.game_tick_number,
       pat.is_airborne,
       warmup,
       lag(pat.view_x) over (partition by round_id order by player_id, tick_id) as last_view_x
from ticks t
         join player_at_tick pat on t.id = pat.tick_id
         join rounds r on t.round_id = r.id
         join games g on r.game_id = g.id
where game_id < 5
limit 100;
