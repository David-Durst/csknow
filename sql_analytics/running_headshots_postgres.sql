select *
from (
         select base_tick_id,
                tick_id,
                player_id,
                pos_x,
                pos_y,
                pos_z,
                leadx,
                lagx,
                lead_tick,
                lag_tick,
                sqrt(pow(leadx - lagx, 2) + pow(leady - lagy, 2) + pow(leadz - lagz, 2)) /
                NULLIF((lead_game_tick - lag_game_tick),0) * game_tick_rate as velocity,
                game_tick_rate
         from (
                  select h.tick_id                                        as base_tick_id,
                         pat.tick_id                                      as tick_id,
                         pat.player_id,
                         h.hit_group,
                         pat.pos_x,
                         pat.pos_y,
                         pat.pos_z,
                         g.game_tick_rate,
                         t.demo_tick_number,
                         t.game_tick_number,
                         pat.is_airborne,
                         warmup,
                         lead(pos_x) over (order by pat.tick_id)              as leadx,
                         lag(pos_x) over (order by pat.tick_id)               as lagx,
                         lead(pos_y) over (order by pat.tick_id)              as leady,
                         lag(pos_y) over (order by pat.tick_id)               as lagy,
                         lead(pos_z) over (order by pat.tick_id)              as leadz,
                         lag(pos_z) over (order by pat.tick_id)               as lagz,
                         lead(t.demo_tick_number) over (order by pat.tick_id) as lead_tick,
                         lag(t.demo_tick_number) over (order by pat.tick_id)  as lag_tick,
                         lead(t.game_tick_number) over (order by pat.tick_id) as lead_game_tick,
                         lag(t.game_tick_number) over (order by pat.tick_id)  as lag_game_tick
                  from hurt h
                           join player_at_tick pat on
                          (pat.tick_id between h.tick_id - 1 and h.tick_id + 1) and h.attacker = pat.player_id
                           join ticks t on pat.tick_id = t.id
                           join rounds r on t.round_id = r.id
                           join games g on r.game_id = g.id
                  where hit_group = 1
                    and warmup = false
                    and not is_airborne
                  order by base_tick_id
              ) i1
         ) i2
where base_tick_id = tick_id and velocity > 170;

set enable_hashjoin = off;
set enable_mergejoin = off;
set enable_hashjoin = on;
set enable_mergejoin = on;
set join_collapse_limit = 1;
show default_statistics_target;
set default_statistics_target = 10000;

