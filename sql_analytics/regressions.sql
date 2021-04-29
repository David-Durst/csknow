USE csknow;

# running headshots
drop table velocity_hs;
create temporary table velocity_hs (
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
                (lead_tick - lag_tick) * game_tick_rate as velocity,
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
                    lead(pos_x) over (order by tick_id)              as leadx,
                    lag(pos_x) over (order by tick_id)               as lagx,
                    lead(pos_y) over (order by tick_id)              as leady,
                    lag(pos_y) over (order by tick_id)               as lagy,
                    lead(pos_z) over (order by tick_id)              as leadz,
                    lag(pos_z) over (order by tick_id)               as lagz,
                    lead(t.demo_tick_number) over (order by tick_id) as lead_tick,
                    lag(t.demo_tick_number) over (order by tick_id)  as lag_tick
             from hurt h
                      join player_at_tick pat on
                     h.tick_id - 1 <= pat.tick_id and h.tick_id + 1 >= pat.tick_id and h.attacker = pat.player_id
                      join ticks t on pat.tick_id = t.id
                      join rounds r on t.round_id = r.id
                      join games g on r.game_id = g.id
             where hit_group = 1
               and warmup = 0
             order by tick_id
         ) innerquery
    where not is_airborne
);

# https://steamcommunity.com/sharedfiles/filedetails/?id=501419345
select * from velocity_hs where base_tick_id = tick_id and velocity > 170;

