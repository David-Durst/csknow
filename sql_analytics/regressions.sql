USE csknow;

select * from hurt h limit 10;

# running headshots
select h.id hid, * from hurt h
                            join player_at_tick pat on
                h.tick_id-1 <= pat.tick_id and h.tick_id+1 >= pat.tick_id and h.attacker = pat.player_id
where hit_group = 1 limit 10;

select base_tick_id, tick_id, player_id, pos_x, pos_y, pos_z, leadx, lagx, lead_tick, lag_tick,
       sqrt(pow(leadx - lagx, 2) + pow(leady - lagy, 2) + pow(leadz - lagz, 2)) / (lead_tick - lag_tick) as velocity
from (
    select h.tick_id as base_tick_id, pat.tick_id as tick_id, pat.player_id, h.hit_group, pat.pos_x, pat.pos_y,
           pat.pos_z, g.game_tick_rate, t.demo_tick_number, pat.is_airborne,
           lead(pos_x) over (order by tick_id) as leadx, lag(pos_x) over (order by tick_id) as lagx,
           lead(pos_y) over (order by tick_id) as leady, lag(pos_y) over (order by tick_id) as lagy,
           lead(pos_z) over (order by tick_id) as leadz, lag(pos_z) over (order by tick_id) as lagz,
           lead(t.demo_tick_number) over (order by tick_id) as lead_tick, lag(t.demo_tick_number) over (order by tick_id) as lag_tick
    from hurt h
        join player_at_tick pat on
            h.tick_id-1 <= pat.tick_id and h.tick_id+1 >= pat.tick_id and h.attacker = pat.player_id
        join ticks t on pat.tick_id = t.id
        join rounds r on t.round_id = r.id
        join games g on r.game_id = g.id
    where hit_group = 1
    order by tick_id
    ) innerquery
where not is_airborne  and base_tick_id = tick_id
limit 10;


select * from hurt where tick_id = 1371204 limit 10;

select * from player_at_tick where tick_id = 1371204 limit 10;
