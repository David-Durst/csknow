select killer, round_id, count(*) as kills_per_round
from kills k
    join ticks t on k.tick_id = t.id
    join rounds r on r.id = t.round_id
where not r.warmup
group by killer, round_id
order by kills_per_round desc
