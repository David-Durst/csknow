select *,
       case when delta_y = 0 and delta_x = 0 then 0
            when abs(move_angle - look_angle) < 22.5 then 1
            else 0 end as forward_backward,
       case when delta_y = 0 and delta_x = 0 then 0
            when abs(move_angle - look_angle) > 22.5 then 1
            else 0 end as strafe
from (
     select tick_id,
            round_id,
            player_id,
            view_x,
            delta_y,
            delta_x,
            mod(degrees(atan(delta_y, delta_x)) + 360, 45) as move_angle,
            mod(view_x + 360, 45) as look_angle
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
                       where g.id < 5
                   ) i1
          ) i2
     ) i3;
