drop table if exists round_start_end_tick;
create temp table round_start_end_tick as
select r.game_id as game_id,
       r.id      as round_id,
       min(t.id) as min_tick_id,
       max(t.id) as max_tick_id
from ticks t
         join rounds r on t.round_id = r.id
group by r.game_id, r.id
order by r.game_id, r.id;