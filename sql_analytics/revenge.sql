select start_round_id, k1_victim, t1_id, revenge_starter, tick_next_round, k3.killer as revenge_ender, k3.victim as revenge_starter_as_victim
from (
         select min(t1.round_id) as start_round_id, k1.victim as k1_victim, t1.id as t1_id, min(k1.killer) as revenge_starter, min(t2.id) as tick_next_round
         from kills k1
                  join ticks t1 on k1.tick_id = t1.id
                  join kills k2 on k1.victim = k2.killer
                  join ticks t2 on t2.id = k2.tick_id
         where t1.round_id = t2.round_id - 1
         group by k1.victim, t1.id, k1.killer
     ) i1
    join kills k3 on k3.tick_id = tick_next_round
where revenge_starter = k3.victim
