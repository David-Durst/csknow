select round_id, player_id, zigging_or_zagging_predicate, min(tick_id) as start_tick, max(tick_id) as end_tick,
       bit_or(strafe_right) + bit_or(strafe_left) as zig_and_zag, count(*) as zig_zag_length
from (
         select *, sum(case when not (forward and (strafe_left xor strafe_right)) then 0 else 1 end)
                       over (partition by round_id, player_id order by tick_id) as zigging_or_zagging_predicate
         from (
                  select *,
                         case when delta_y = 0 and delta_x = 0 then 0
                              when abs(move_angle - look_angle) < 45 or abs(move_angle - look_angle) > 315 and is_alive then 1
                              else 0 end as forward,
                         case when delta_y = 0 and delta_x = 0 then 0
                              when abs(move_angle - look_angle) > 45 and abs(move_angle - look_angle) < 135 and is_alive then 1
                              else 0 end as strafe_right,
                         case when delta_y = 0 and delta_x = 0 then 0
                              when abs(move_angle - look_angle) > 135 and abs(move_angle - look_angle) < 225 and is_alive then 1
                              else 0 end as backward,
                         case when delta_y = 0 and delta_x = 0 then 0
                              when abs(move_angle - look_angle) > 225 and abs(move_angle - look_angle) < 315 and is_alive then 1
                              else 0 end as strafe_left
                  from (
                           select tick_id,
                                  round_id,
                                  player_id,
                                  is_alive,
                                  view_x,
                                  delta_y,
                                  delta_x,
                                  demo_tick_rate,
                                  mod(degrees(atan(delta_y, delta_x)) + 360, 360) as move_angle,
                                  mod(view_x + 360, 360) as look_angle
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
                                                    g.demo_tick_rate,
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
     ) i5
group by round_id, player_id, zigging_or_zagging_predicate
having zig_and_zag = 2 and zig_zag_length > max(demo_tick_rate)
