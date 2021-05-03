select sum(moved) / sum(case when is_alive then 1 else 0 end) as percent_moved,
       sum(is_killer) /  sum(is_victim) as kd,
       sum(is_killer) as num_kills,
       sum(is_victim) as num_deaths,
       sum(is_killer) / (sum(moved) / sum(case when is_alive then 1 else 0 end)) as kills_per_movement,
       player_id
from (
         select *,
                case when (delta_y != 0 or delta_x != 0) and is_alive then 1
                     else 0 end as moved,
                case when killer = player_id then 1 else 0 end as is_killer,
                case when victim = player_id then 1 else 0 end as is_victim
         from (
                  select tick_id,
                         round_id,
                         player_id,
                         is_alive,
                         killer,
                         victim,
                         view_x,
                         delta_y,
                         delta_x
                  from (
                           select *,
                                  pos_x - lagx as delta_x,
                                  pos_y - lagy as delta_y
                           from (
                                    select pat.tick_id,
                                           pat.player_id,
                                           r.id as round_id,
                                           pat.pos_x,
                                           pat.pos_y,
                                           pat.view_x,
                                           pat.is_alive,
                                           g.game_tick_rate,
                                           t.demo_tick_number,
                                           t.game_tick_number,
                                           warmup,
                                           killer,
                                           victim,
                                           lag(pos_x) over (partition by round_id, player_id order by tick_id)               as lagx,
                                           lag(pos_y) over (partition by round_id, player_id order by tick_id)               as lagy,
                                           lag(t.demo_tick_number) over (partition by round_id, player_id order by tick_id)  as lag_tick
                                    from player_at_tick pat
                                             join ticks t on pat.tick_id = t.id
                                             join rounds r on t.round_id = r.id
                                             join games g on r.game_id = g.id
                                             left join kills k on t.id = k.tick_id and (pat.player_id = k.killer or pat.player_id = k.victim)
                                    where g.id < 5 and player_id < 60 # 60 is a hack, some players from later games being assigned to game 0 during init, they don't move, so ignore them
                                ) i1
                       ) i2
              ) i3
     ) i4
group by player_id
order by kills_per_movement desc
;

