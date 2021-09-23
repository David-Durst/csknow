drop table last_spotted_cpu;
create temp table last_spotted_cpu as
select *,
       case when not is_spotted then first_value(lag_game_tick_number)
           over (partition by r_id, spotter_player, spotted_player, last_spotted_region_id order by tick_id)
           else game_tick_number end as last_spotted_game_tick_number
from (
         select *,
                sum(not_spotted_indicator)
                over (partition by r_id, spotter_player, spotted_player order by tick_id)
                    as last_spotted_region_id
         from (
                  select *,
                         -- use lag_is_spotted here as guaranteed
                         case when lag_is_spotted and not is_spotted then 1 else 0 end as not_spotted_indicator
                  from (
                           select g.id as g_id,
                                  r.id as r_id,
                                  t.id as tick_id,
                                  s.spotted_player,
                                  s.spotter_player,
                                  t.game_tick_number,
                                  s.is_spotted,
                                  lag(t.game_tick_number, 1, cast(0 as bigint))
                                  over (partition by r.id, s.spotter_player, s.spotted_player order by tick_id)
                                       as lag_game_tick_number,
                                  lag(s.is_spotted, 1, false)
                                  over (partition by r.id, s.spotter_player, s.spotted_player order by tick_id)
                                       as lag_is_spotted
                           from spotted s
                                    join ticks t on s.tick_id = t.id
                                    join rounds r on t.round_id = r.id
                                    join games g on r.game_id = g.id
                           order by t.id, s.id
                       ) as t_1
              ) as t_2
     ) as t_3;


drop table hand_visibility_with_next_start;
create temp table hand_visibility_with_next_start as
select *,
       lead(start_game_tick, 1, cast(1e7 as bigint))
           over (partition by demo, spotter_id, spotted_id order by start_game_tick)
           as next_start_game_tick
from hand_visibility;


drop table visibilities;
create temp table visibilities as
select h.index,
       h.demo,
       min(tick_id)            as tick_id,
       h.spotter_id,
       h.spotted_id,
       h.spotted,
       h.start_game_tick,
       h.end_game_tick,
       h.next_start_game_tick,
       min(s.game_tick_number) as automated_vis_tick,
       h.hacking
from last_spotted_cpu s
         right join hand_visibility_with_next_start h
                    on h.spotted_id = s.spotted_player
                        and h.spotter_id = s.spotter_player
                        and h.start_game_tick <= s.game_tick_number
                        and h.next_start_game_tick >= s.game_tick_number
                        and s.is_spotted = true
group by h.index, h.demo, h.spotter_id, h.spotted_id, h.spotted, h.start_game_tick, h.end_game_tick,
         h.next_start_game_tick, h.hacking
order by h.index;


-- this table and next row shows that for me, only missed CPU visibility events from above join
-- (aka null values for hand_visibility_with_next_start) are in warmup, so join is ok
drop table ungrouped_visibilities;
create temp table ungrouped_visibilities as
select h.index, h.demo, s.tick_id, h.spotter_id, h.spotted_id, h.spotted, h.start_game_tick, h.end_game_tick, h.next_start_game_tick, s.game_tick_number,
       s.is_spotted, s.spotted_player as s_spotted_player, s.spotter_player as s_spotter_player, s.g_id
from last_spotted_cpu s
         full join hand_visibility_with_next_start h
                   on h.spotted_id = s.spotted_player
                       and h.spotter_id = s.spotter_player
                       and h.start_game_tick <= s.game_tick_number
                       and h.next_start_game_tick >= s.game_tick_number
                       and s.is_spotted = true
order by h.index;

select g.demo_file, game_tick_number, is_spotted, p_spotter.name as spotter, p_spotted.name as spotted
from ungrouped_visibilities
    join players p_spotted on p_spotted.id = s_spotted_player
    join players p_spotter on p_spotter.id = s_spotter_player
    join player_at_tick pat_spotted on p_spotted.id = pat_spotted.player_id and pat_spotted.tick_id = ungrouped_visibilities.tick_id
    join player_at_tick pat_spotter on p_spotter.id = pat_spotter.player_id and pat_spotter.tick_id = ungrouped_visibilities.tick_id
    join games g on p_spotted.game_id = g.id
where spotted is null
    and is_spotted and p_spotter.name in ('i_eat_short_people_for_breakfast')
    and pat_spotted.is_alive and pat_spotter.is_alive and pat_spotter.team != pat_spotted.team
order by  demo_file, game_tick_number;



drop table react_ticks;
create temp table react_ticks as
select v.index, v.demo, v.tick_id, v.spotter_id, v.spotted_id, v.spotted, v.start_game_tick, v.end_game_tick, v.next_start_game_tick, v.automated_vis_tick, min(t.game_tick_number) as react_end_tick, v.hacking
from lookers l
         join ticks t on l.tick_id = t.id
         right join visibilities v
                    on v.start_game_tick <= t.game_tick_number
                        and v.next_start_game_tick >= t.game_tick_number
                        and l.looker_player_id = v.spotter_id
                        and l.looked_at_player_id = v.spotted_id
group by v.index, v.demo, v.tick_id, v.spotter_id, v.spotted_id, v.spotted, v.start_game_tick, v.end_game_tick, v.next_start_game_tick, v.automated_vis_tick, v.hacking
order by v.index;

drop table react_final;
create temp table react_final as
select *, (react_end_tick - start_game_tick) / 64.0 as hand_react_ms, (react_end_tick - react_ticks.automated_vis_tick) / 64.0 as automated_react_ms
from react_ticks where (react_end_tick - start_game_tick) / 64.0 < 3;

select * from react_final;