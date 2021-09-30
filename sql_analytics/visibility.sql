drop table if exists last_spotted_cpu;
create temp table last_spotted_cpu as
select *,
       case
           when not is_spotted then first_value(lag_game_tick_number)
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


drop table if exists hand_visibility_with_next_start;
create temp table hand_visibility_with_next_start as
select *,
       lead(start_game_tick, 1, cast(1e7 as bigint))
       over (partition by demo, spotter_id, spotted_id order by start_game_tick)
           as next_start_game_tick
from hand_visibility;


drop table if exists visibilities;
create temp table visibilities as
select h.index,
       h.demo,
       min(s.tick_id)          as cpu_tick_id,
       h.spotter_id,
       h.spotter,
       h.spotted_id,
       h.spotted,
       h.start_game_tick,
       h.end_game_tick,
       h.next_start_game_tick,
       min(s.game_tick_number) as cpu_vis_game_tick,
       h.hacking
from last_spotted_cpu s
         right join hand_visibility_with_next_start h
                    on h.spotted_id = s.spotted_player
                        and h.spotter_id = s.spotter_player
                        and h.start_game_tick <= s.game_tick_number
                        and h.next_start_game_tick >= s.game_tick_number
                        and s.is_spotted = true
group by h.index, h.demo, h.spotter_id, h.spotter, h.spotted_id, h.spotted, h.start_game_tick, h.end_game_tick,
         h.next_start_game_tick, h.hacking
order by h.index;


-- this table and next row shows that for me, only missed CPU visibility events from above join
-- (aka null values for hand_visibility_with_next_start) are in warmup, so join is ok
drop table if exists ungrouped_visibilities;
create temp table ungrouped_visibilities as
select h.index,
       h.demo,
       s.tick_id,
       h.spotter_id,
       h.spotted_id,
       h.spotted,
       h.start_game_tick,
       h.end_game_tick,
       h.next_start_game_tick,
       s.game_tick_number,
       s.is_spotted,
       s.spotted_player as s_spotted_player,
       s.spotter_player as s_spotter_player,
       s.g_id
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
         join player_at_tick pat_spotted
              on p_spotted.id = pat_spotted.player_id and pat_spotted.tick_id = ungrouped_visibilities.tick_id
         join player_at_tick pat_spotter
              on p_spotter.id = pat_spotter.player_id and pat_spotter.tick_id = ungrouped_visibilities.tick_id
         join games g on p_spotted.game_id = g.id
where spotted is null
  and is_spotted
  and p_spotter.name in ('i_eat_short_people_for_breakfast')
  and pat_spotted.is_alive
  and pat_spotter.is_alive
  and pat_spotter.team != pat_spotted.team
order by demo_file, game_tick_number;


drop table if exists visibilities_with_others;
create temp table visibilities_with_others as
select v_main.index,
       v_main.demo,
       v_main.cpu_tick_id,
       v_main.spotter_id,
       v_main.spotter,
       v_main.spotted_id,
       v_main.spotted,
       v_main.start_game_tick,
       v_main.end_game_tick,
       v_main.next_start_game_tick,
       v_main.cpu_vis_game_tick,
       v_main.hacking,
       count(distinct v_other.spotted_id) as distinct_others_spotted_during_time
from visibilities v_main
         left join visibilities v_other
                   on v_main.spotter_id = v_other.spotter_id
                       and v_main.spotted_id != v_other.spotted_id
                       and int8range(v_main.start_game_tick, v_main.end_game_tick) &&
                           int8range(v_other.start_game_tick, v_other.end_game_tick)
group by v_main.index,
         v_main.demo,
         v_main.cpu_tick_id,
         v_main.spotter_id,
         v_main.spotter,
         v_main.spotted_id,
         v_main.spotted,
         v_main.start_game_tick,
         v_main.end_game_tick,
         v_main.next_start_game_tick,
         v_main.cpu_vis_game_tick,
         v_main.hacking
order by v_main.index;


drop table if exists react_aim_ticks;
create temp table react_aim_ticks as
select v.index,
       v.demo,
       v.cpu_tick_id,
       v.spotter_id,
       v.spotter,
       v.spotted_id,
       v.spotted,
       v.start_game_tick,
       v.end_game_tick,
       v.next_start_game_tick,
       v.cpu_vis_game_tick,
       min(t.game_tick_number) as react_aim_end_tick,
       v.distinct_others_spotted_during_time,
       v.hacking
from lookers l
         join ticks t on l.tick_id = t.id
         right join visibilities_with_others v
                    on v.start_game_tick - 128 <= t.game_tick_number
                        and v.next_start_game_tick >= t.game_tick_number
                        and l.looker_player_id = v.spotter_id
                        and l.looked_at_player_id = v.spotted_id
group by v.index, v.demo, v.cpu_tick_id, v.spotter_id, v.spotter, v.spotted_id, v.spotted, v.start_game_tick,
         v.end_game_tick, v.next_start_game_tick, v.cpu_vis_game_tick, v.distinct_others_spotted_during_time, v.hacking
order by v.index;


drop table if exists react_aim_and_fire_ticks;
create temp table react_aim_and_fire_ticks as
select rat.index,
       rat.demo,
       rat.cpu_tick_id,
       rat.spotter_id,
       rat.spotter,
       rat.spotted_id,
       rat.spotted,
       rat.start_game_tick,
       rat.end_game_tick,
       rat.next_start_game_tick,
       rat.cpu_vis_game_tick,
       rat.react_aim_end_tick,
       min(t.game_tick_number) as react_fire_end_tick,
       rat.distinct_others_spotted_during_time,
       rat.hacking
from hurt h
         join ticks t on h.tick_id = t.id
         right join react_aim_ticks rat
                    on rat.start_game_tick - 128 <= t.game_tick_number
                        and rat.next_start_game_tick >= t.game_tick_number
                        and h.attacker = rat.spotter_id
                        and h.victim = rat.spotted_id
group by rat.index, rat.demo, rat.cpu_tick_id, rat.spotter_id, rat.spotter, rat.spotted_id, rat.spotted, rat.start_game_tick,
         rat.end_game_tick, rat.next_start_game_tick, rat.cpu_vis_game_tick, rat.react_aim_end_tick, rat.distinct_others_spotted_during_time, rat.hacking
order by rat.index;


drop table if exists round_start_end_tick;
create temp table round_start_end_tick as
select r.game_id               as game_id,
       r.id                    as round_id,
       min(t.id)               as min_tick_id,
       max(t.id)               as max_tick_id,
       min(t.game_tick_number) as min_game_tick,
       max(t.game_tick_number) as max_game_tick
from ticks t
         join rounds r on t.round_id = r.id
group by r.game_id, r.id
order by r.game_id, r.id;


drop table if exists react_final;
create temp table react_final as
select raft.index,
       raft.demo,
       raft.cpu_tick_id,
       raft.spotter_id,
       raft.spotter,
       raft.spotted_id,
       raft.spotted,
       raft.start_game_tick,
       raft.end_game_tick,
       raft.next_start_game_tick,
       raft.cpu_vis_game_tick,
       raft.react_aim_end_tick,
       raft.react_fire_end_tick,
       raft.distinct_others_spotted_during_time,
       raft.hacking,
       (raft.react_aim_end_tick - raft.start_game_tick) / 64.0   as hand_aim_react_s,
       (raft.react_aim_end_tick - raft.cpu_vis_game_tick) / 64.0 as cpu_aim_react_s,
       (raft.react_fire_end_tick - raft.start_game_tick) / 64.0   as hand_fire_react_s,
       (raft.react_fire_end_tick - raft.cpu_vis_game_tick) / 64.0 as cpu_fire_react_s,
       rset.round_id,
       rset.game_id
from react_aim_and_fire_ticks raft
         join games g on g.demo_file = raft.demo
         join round_start_end_tick rset
              on g.id = rset.game_id
                  and int8range(raft.start_game_tick, raft.end_game_tick) &&
                      int8range(rset.min_game_tick, rset.max_game_tick);
