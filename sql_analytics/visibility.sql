drop table if exists hacking_per_game;
create temp table hacking_per_game as
select distinct demo, hacking from visibilities;

drop table if exists visibility_sources;
create temp table visibility_sources as
select * from (values (0, 'pixel_adjusted'), (1, 'pixel_unadjusted'), (2, 'bbox')) AS t (id,visibility_technique);

-- per round, sequence of ticks with repeated true for still spotted and false for still not spotted
-- so find the transitions from spotted to not spotted, create counter separating spotted regions
-- and then within each region get the start and stopping of when being spotted
drop table if exists visibilities_bbox;
create temp table visibilities_bbox as
select demo,
       spotter_id,
       spotter,
       spotted_id,
       spotted,
       min(game_tick_number) as start_game_tick,
       max(lead_game_tick_number) as end_game_tick,
       hacking
from (
         select *
         from (
                  select *,
                         sum(not_spotted_indicator)
                         over (partition by r_id, spotter_id, spotted_id order by tick_id)
                             as spotted_region_id
                  from (
                           select *,
                                  -- use lag_is_spotted here as guaranteed
                                  case when lag_is_spotted and not is_spotted then 1 else 0 end as not_spotted_indicator
                           from (
                                    select g.demo_file        as demo,
                                           s.spotter_player   as spotter_id,
                                           pspotter.name      as spotter,
                                           s.spotted_player   as spotted_id,
                                           pspotted.name      as spotted,
                                           hpg.hacking        as hacking,
                                           g.id               as g_id,
                                           r.id               as r_id,
                                           t.id               as tick_id,
                                           t.game_tick_number as game_tick_number,
                                           s.is_spotted       as is_spotted,
                                           lead(t.game_tick_number, 1)
                                           over (partition by r.id, s.spotter_player, s.spotted_player order by tick_id)
                                               as lead_game_tick_number,
                                           lag(s.is_spotted, 1, false)
                                           over (partition by r.id, s.spotter_player, s.spotted_player order by tick_id)
                                                              as lag_is_spotted
                                    from spotted s
                                             join ticks t on s.tick_id = t.id
                                             join rounds r on t.round_id = r.id
                                             join games g on r.game_id = g.id
                                             join players pspotted on s.spotted_player = pspotted.id
                                             join players pspotter on s.spotter_player = pspotter.id
                                             join hacking_per_game hpg on g.demo_file = hpg.demo
                                    order by r.id, spotter_id, spotted_id, tick_id
                                ) as t_1
                       ) as t_2
              ) as t_3
         where is_spotted = True
     ) t_4
group by demo, spotter_id, spotter, spotted_id, spotted, hacking, g_id, r_id;

drop table if exists all_visibilities;
create temp table all_visibilities as
select demo,
       spotter_id,
       spotter,
       spotted_id,
       spotted,
       start_game_tick,
       end_game_tick,
       hacking,
       0 as visibility_technique_id from visibilities
union
select demo,
       spotter_id,
       spotter,
       spotted_id,
       spotted,
       start_game_tick,
       end_game_tick,
       hacking,
       1 as visibility_technique_id from visibilities_unadjusted
union
select demo,
       spotter_id,
       spotter,
       spotted_id,
       spotted,
       start_game_tick,
       end_game_tick,
       hacking,
       2 as visibility_technique_id from visibilities_bbox;

drop table if exists visibilities_with_next_start;
create temp table visibilities_with_next_start as
select *,
       lead(start_game_tick, 1, cast(1e7 as bigint))
       over (partition by visibility_technique_id, demo, spotter_id, spotted_id order by start_game_tick)
           as next_start_game_tick,
       lag(end_game_tick, 1, cast(0 as bigint))
       over (partition by visibility_technique_id, demo, spotter_id, spotted_id order by start_game_tick)
           as last_end_game_tick
from all_visibilities;



drop table if exists visibilities_with_others;
create temp table visibilities_with_others as
select v_main.demo,
       v_main.visibility_technique_id,
       v_main.spotter_id,
       v_main.spotter,
       v_main.spotted_id,
       v_main.spotted,
       v_main.start_game_tick,
       v_main.end_game_tick,
       v_main.next_start_game_tick,
       v_main.last_end_game_tick,
       v_main.hacking,
       count(distinct v_other.spotted_id) as distinct_others_spotted_during_time
from visibilities_with_next_start v_main
         left join visibilities_with_next_start v_other
                   on v_main.spotter_id = v_other.spotter_id
                       and v_main.visibility_technique_id = v_main.visibility_technique_id
                       and v_main.spotted_id != v_other.spotted_id
                       and int8range(v_main.start_game_tick, v_main.end_game_tick) &&
                           int8range(v_other.start_game_tick, v_other.end_game_tick)
group by v_main.demo,
         v_main.visibility_technique_id,
         v_main.spotter_id,
         v_main.spotter,
         v_main.spotted_id,
         v_main.spotted,
         v_main.start_game_tick,
         v_main.end_game_tick,
         v_main.next_start_game_tick,
         v_main.last_end_game_tick,
         v_main.hacking
order by v_main.demo, v_main.start_game_tick, v_main.spotter, v_main.spotted;


drop table if exists lookers_with_last;
create temp table lookers_with_last as
select *,
       last_value(seconds_since_look_started)
       over (partition by game_id, round_id, looker_player_id, looked_at_player_id, look_id order by tick_id  RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
           as look_length
    from (
         select *,
                sum(seconds_since_last_looker_row_per_look)
                over (partition by game_id, round_id, looker_player_id, looked_at_player_id, look_id order by tick_id)
                    as seconds_since_look_started
         from (
                  select *,
                         sum(new_look)
                         over (partition by game_id, round_id, looker_player_id, looked_at_player_id order by tick_id)
                             as look_id,
                         case when new_look = 1 then 0 else seconds_since_last_looker_row end
                             as seconds_since_last_looker_row_per_look
                  from (
                           select *,
                                  case when seconds_since_last_looker_row > 0.1 then 1 else 0 end
                                      as new_look
                           from (
                                    select *,
                                           (game_tick_number - last_look_game_tick) /
                                           cast(game_tick_rate as double precision)
                                               as seconds_since_last_looker_row
                                    from (
                                             select *,
                                                    lag(game_tick_number, 1, cast(0 as bigint))
                                                    over (partition by game_id, round_id, looker_player_id, looked_at_player_id order by tick_id)
                                                        as last_look_game_tick
                                             from (
                                                      select l.index,
                                                             l.tick_id,
                                                             l.looker_pat_id,
                                                             l.looker_player_id,
                                                             l.looked_at_pat_id,
                                                             l.looked_at_player_id,
                                                             t.game_tick_number,
                                                             g.game_tick_rate,
                                                             g.id as game_id,
                                                             r.id as round_id
                                                      from lookers l
                                                               join ticks t on l.tick_id = t.id
                                                               join rounds r on t.round_id = r.id
                                                               join games g on r.game_id = g.id
                                                  ) inner1
                                         ) inner2
                                ) inner3
                       ) inner4
              ) inner5
     ) inner6;

select * from lookers_with_last;

drop table if exists react_aim_ticks;
create temp table react_aim_ticks as
select v.demo,
       v.visibility_technique_id,
       v.spotter_id,
       v.spotter,
       v.spotted_id,
       v.spotted,
       v.start_game_tick,
       v.end_game_tick,
       v.next_start_game_tick,
       v.last_end_game_tick,
       min(t.game_tick_number) as react_aim_end_tick,
       v.distinct_others_spotted_during_time,
       v.hacking
from (select * from lookers_with_last where new_look = 1 and look_length >= 0.1) l
         join ticks t on l.tick_id = t.id
         right join visibilities_with_others v
                    on v.start_game_tick - (64*3) <= t.game_tick_number
                        and v.next_start_game_tick >= t.game_tick_number
                        and l.looker_player_id = v.spotter_id
                        and l.looked_at_player_id = v.spotted_id
group by v.demo, v.visibility_technique_id, v.spotter_id, v.spotter, v.spotted_id, v.spotted, v.start_game_tick,
         v.end_game_tick, v.next_start_game_tick, v.last_end_game_tick, v.distinct_others_spotted_during_time, v.hacking
order by v.demo, v.start_game_tick, v.spotter, v.spotted;


drop table if exists react_aim_and_fire_ticks;
create temp table react_aim_and_fire_ticks as
select rat.demo,
       rat.visibility_technique_id,
       rat.spotter_id,
       rat.spotter,
       rat.spotted_id,
       rat.spotted,
       rat.start_game_tick,
       rat.end_game_tick,
       rat.next_start_game_tick,
       rat.last_end_game_tick,
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
group by rat.demo, rat.visibility_technique_id, rat.spotter_id, rat.spotter, rat.spotted_id, rat.spotted, rat.start_game_tick,
         rat.end_game_tick, rat.next_start_game_tick, rat.react_aim_end_tick, rat.last_end_game_tick, rat.distinct_others_spotted_during_time, rat.hacking
order by rat.demo, rat.start_game_tick, rat.spotter, rat.spotted;


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
select raft.demo,
       raft.visibility_technique_id,
       raft.spotter_id,
       raft.spotter,
       raft.spotted_id,
       raft.spotted,
       raft.start_game_tick,
       raft.end_game_tick,
       raft.next_start_game_tick,
       raft.last_end_game_tick,
       (raft.start_game_tick - raft.last_end_game_tick) / cast(g.game_tick_rate as double precision) < 5 as seen_last_five_seconds,
       raft.react_aim_end_tick,
       raft.react_fire_end_tick,
       raft.distinct_others_spotted_during_time,
       raft.hacking,
       (raft.react_aim_end_tick - raft.start_game_tick) / cast(g.game_tick_rate as double precision) as aim_react_s,
       (raft.react_fire_end_tick - raft.start_game_tick) / cast(g.game_tick_rate as double precision) as fire_react_s,
       rset.round_id,
       rset.game_id
from react_aim_and_fire_ticks raft
         join games g on g.demo_file = raft.demo
         join round_start_end_tick rset
              on g.id = rset.game_id
                  and int8range(raft.start_game_tick, raft.end_game_tick) &&
                      int8range(rset.min_game_tick, rset.max_game_tick);
