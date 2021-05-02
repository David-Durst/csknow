select pat.player_id,
       round_id,
       min(p.name),
       sum(case when pat.is_crouching and pat.is_alive then 1 else 0 end) / count(case when pat.is_alive then 1 else 0 end) as pct_crouching
from ticks t
         join player_at_tick pat on t.id = pat.tick_id
         join rounds r on t.round_id = r.id
         join games g on r.game_id = g.id
         join players p on pat.player_id = p.id
where not warmup
group by round_id, pat.player_id
order by pct_crouching DESC;

select * from rounds join games g on rounds.game_id = g.id join ticks t on rounds.start_tick = t.id where rounds.id = 965