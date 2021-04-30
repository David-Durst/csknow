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
       lag(t.demo_tick_number) over (order by pat.tick_id)  as lag_tick
from ticks t
         join player_at_tick pat on
                t.id = pat.tick_id - 1 or
                t.id = pat.tick_id + 1 or
                t.id = pat.tick_id
         join hurt h on h.tick_id = t.id and h.attacker = pat.player_id
         join rounds r on t.round_id = r.id
         join games g on r.game_id = g.id
where hit_group = 1
  and warmup = false
order by tick_id
