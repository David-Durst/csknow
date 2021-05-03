select round_id, player_id, count(*) as length_of_movement
from
     (
     select *, sum(case when moved = 0 then 1 else 0 end) over (partition by round_id, player_id order by tick_id) as continuous_movement
     from (
              select *,
                     case when (delta_y != 0 or delta_x != 0) and is_alive then 1
                          else 0 end as moved
              from (
                       select tick_id,
                              round_id,
                              player_id,
                              is_alive,
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
                                       lag(pos_x) over (partition by round_id, player_id order by tick_id)               as lagx,
                                       lag(pos_y) over (partition by round_id, player_id order by tick_id)               as lagy,
                                       lag(t.demo_tick_number) over (partition by round_id, player_id order by tick_id)  as lag_tick
                                from player_at_tick pat
                                         join ticks t on pat.tick_id = t.id
                                         join rounds r on t.round_id = r.id
                                         join games g on r.game_id = g.id
                                where g.id < 5 and player_id < 60 # 60 is a hack, some players from later games being assigned to game 0 during init, they don't move, so ignore them
                            ) i1
                   ) i2
          ) i3
     ) i4
where is_alive
group by round_id, player_id, continuous_movement
order by length_of_movement desc
;

