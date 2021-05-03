# this fails, see below for why
select hurt_tick_id,
       round_id,
       game_id,
       min(game_tick_number)
       #sum(forward), sum(backward), sum(strafe)
       #sum(forward) / sum(case when is_alive then 1 else 0 end) as percent_forward,
       #sum(backward) / sum(case when is_alive then 1 else 0 end) as percent_backward,
       #sum(strafe) / sum(case when is_alive then 1 else 0 end) as percent_strafe,
       #sum(view_delta) as amount_turn,
       #sum(case when is_alive and is_airborne then 1 else 0 end) /
       #     sum(case when is_alive then 1 else 0 end) as percent_airborne,
       #sum(case when is_alive and is_crouching then 1 else 0 end) /
       #    sum(case when is_alive then 1 else 0 end) as percent_crouching
from (
         select *,
                case
                    when delta_y = 0 and delta_x = 0 then 0
                    when abs(move_angle - look_angle) < 45 and is_alive then 1
                    else 0 end as forward,
                case
                    when delta_y = 0 and delta_x = 0 then 0
                    when abs(move_angle - look_angle) > 45 and abs(move_angle - look_angle) < 135 and is_alive then 1
                    else 0 end as strafe,
                 case
                        when delta_y = 0 and delta_x = 0 then 0
                        when abs(move_angle - look_angle) > 135 and is_alive then 1
                        else 0 end as backward
         from (
                  select *,
                         mod(degrees(atan(delta_y, delta_x)) + 360, 180) as move_angle,
                         mod(view_x + 360, 180)                          as look_angle
                  from (
                           select *,
                                  pos_x - lagx as delta_x,
                                  pos_y - lagy as delta_y,
                                  case
                                      when not is_alive then 0
                                      when view_x > 350 and last_view_x < 10 then -1 * (last_view_x + 360 - view_x)
                                      when last_view_x > 350 and view_x < 10 then view_x + 360 - last_view_x
                                      else view_x - last_view_x end
                                               as view_delta
                           from (
                                    select t.id                                                                     as tick_id,
                                           h.tick_id                                                                as hurt_tick_id,
                                           round_id,
                                           game_id,
                                           game_tick_number,
                                           player_id,
                                           wf.weapon,
                                           pos_x,
                                           pos_y,
                                           pos_z,
                                           view_x,
                                           is_crouching,
                                           is_airborne,
                                           is_alive,
                                           lag(pos_x) over (partition by round_id, player_id order by tick_id)      as lagx,
                                           lag(pos_y) over (partition by round_id, player_id order by tick_id)      as lagy,
                                           lag(pat.view_x) over (partition by round_id, player_id order by tick_id) as last_view_x
                                    from ticks t
                                             join player_at_tick pat on t.id = pat.tick_id
                                             left join weapon_fire wf on pat.tick_id = wf.tick_id and pat.player_id = wf.shooter
                                             join rounds r on t.round_id = r.id
                                             join games g on r.game_id = g.id
                                             join hurt h
                                                  on pat.id >= h.tick_id and pat.player_id <= h.tick_id + 2 * demo_tick_rate and
                                                     h.victim = pat.player_id
                                ) i1
                       ) i2
              ) i3
     ) i4
group by game_id, round_id, hurt_tick_id;

# this fails, join runs out of memory
select t.id                                                                     as tick_id,
       h.tick_id                                                                as hurt_tick_id,
       round_id,
       game_id,
       game_tick_number,
       player_id,
       pos_x,
       pos_y,
       pos_z,
       view_x,
       is_crouching,
       is_airborne,
       is_alive,
       lag(pos_x) over (partition by round_id, player_id order by tick_id)      as lagx,
       lag(pos_y) over (partition by round_id, player_id order by tick_id)      as lagy,
       lag(pat.view_x) over (partition by round_id, player_id order by tick_id) as last_view_x
from ticks t
         join player_at_tick pat on t.id = pat.tick_id
         join rounds r on t.round_id = r.id
         join games g on r.game_id = g.id
         join hurt h
              on pat.id >= h.tick_id and pat.player_id <= h.tick_id + 2 * demo_tick_rate and
                 h.victim = pat.player_id
where game_id < 5;

# this runs because left joining (returning more results, but don't need complex join to produce them)
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
       victim,
       lag(pos_x) over (partition by round_id, player_id order by tick_id)               as lagx,
       lag(pos_y) over (partition by round_id, player_id order by tick_id)               as lagy,
       lag(t.demo_tick_number) over (partition by round_id, player_id order by tick_id)  as lag_tick
from player_at_tick pat
         join ticks t on pat.tick_id = t.id
         join rounds r on t.round_id = r.id
         join games g on r.game_id = g.id
         left join hurt h on pat.id = h.tick_id
where g.id < 5 and player_id < 60 # 60 is a hack, some players from later games being assigned to game 0 during init, they don't move, so ignore them
