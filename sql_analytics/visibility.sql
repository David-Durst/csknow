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

select
    t.id as tick_id,
    t.round_id as round_id,
    t.demo_tick_number as demo_tick_number,
    t.game_tick_number as game_tick_number,
    lead(t.game_tick_number) over (partition by g.id order by t.id) as nex,
    t.bomb_carrier as bomb_carrier,
    t.bomb_x as bomb_x,
    t.bomb_y as bomb_y,
    t.bomb_z as bomb_z,
    r.game_id,
    g.demo_file,
    g.game_tick_rate,
    g.demo_tick_rate
from ticks t
    join rounds r on t.round_id = r.id
    join games g on r.game_id = g.id
order by t.id
;


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

-- clustering has no benefit, just analyze
analyze ticks_rounds_games;
analyze visibilities_with_next_start;

drop table if exists visibilities_with_game_tick;
create temp table visibilities_with_game_tick as
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
       v.hacking,
       trg.game_id,
       trg.round_id,
       trg.tick_id as start_tick_id,
       trg.game_tick_rate
from visibilities_with_next_start v
    join ticks_rounds_games trg
        on v.demo = trg.demo_file
               and trg.game_tick_number <= v.start_game_tick
               and trg.next_game_tick_number > v.start_game_tick
;

drop table if exists visibilities_filtered_alive;
create temp table visibilities_filtered_alive as
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
       v.hacking,
       v.game_id,
       v.round_id,
       v.start_tick_id,
       v.game_tick_rate
from visibilities_with_game_tick v
    join player_at_tick spotter_pat
        on v.start_tick_id = spotter_pat.tick_id
               and spotter_pat.player_id = v.spotter_id
               and spotter_pat.is_alive
where spotter_pat.is_alive;


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
       v_main.game_id,
       v_main.round_id,
       v_main.start_tick_id,
       v_main.game_tick_rate,
       count(distinct v_other.spotted_id) as distinct_others_spotted_during_time
from visibilities_filtered_alive v_main
         left join visibilities_filtered_alive v_other
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
         v_main.hacking,
         v_main.game_id,
         v_main.round_id,
         v_main.start_tick_id,
         v_main.game_tick_rate
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
       v.hacking,
       v.game_id,
       v.round_id,
       v.start_tick_id,
       v.game_tick_rate
from (select * from lookers_with_last where new_look = 1 and look_length >= 0.1) l
         join ticks t on l.tick_id = t.id
         join rounds r on t.round_id = r.id
         join games g on r.game_id = g.id
         right join visibilities_with_others v
                    on v.start_game_tick - (g.game_tick_rate*3) <= t.game_tick_number
                        and v.next_start_game_tick >= t.game_tick_number
                        and l.looker_player_id = v.spotter_id
                        and l.looked_at_player_id = v.spotted_id
group by v.demo, v.visibility_technique_id, v.spotter_id, v.spotter, v.spotted_id, v.spotted, v.start_game_tick,
         v.end_game_tick, v.next_start_game_tick, v.last_end_game_tick, v.distinct_others_spotted_during_time, v.hacking,
         v.game_id, v.round_id, v.start_tick_id, v.game_tick_rate
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
       rat.hacking,
       rat.game_id,
       rat.round_id,
       rat.start_tick_id,
       rat.game_tick_rate
from weapon_fire f
         join ticks t on f.tick_id = t.id
         right join react_aim_ticks rat
                    on rat.start_game_tick <= t.game_tick_number
                        and rat.next_start_game_tick >= t.game_tick_number
                        and f.shooter = rat.spotter_id
group by rat.demo, rat.visibility_technique_id, rat.spotter_id, rat.spotter, rat.spotted_id, rat.spotted, rat.start_game_tick,
         rat.end_game_tick, rat.next_start_game_tick, rat.react_aim_end_tick, rat.last_end_game_tick, rat.distinct_others_spotted_during_time, rat.hacking,
         rat.game_id, rat.round_id, rat.start_tick_id, rat.game_tick_rate
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
where t.game_tick_number > 10
group by r.game_id, r.id
order by r.game_id, r.id;

select count(*) from round_start_end_tick;


drop table if exists cover_origins_with_cluster_counts;
create temp table cover_origins_with_cluster_counts as
select co.index as index,
       co.x as x,
       co.y as y,
       co.z as z,
       count(distinct ce.cluster_id) as num_clusters
from cover_origins co
    join cover_edges ce on ce.origin_id = co.index
group by co.index, co.x, co.y, co.z
order by co.index;


drop table if exists react_aim_fire_cover;
create temp table react_aim_fire_cover as
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
       raft.react_aim_end_tick,
       raft.react_fire_end_tick,
       raft.distinct_others_spotted_during_time,
       raft.hacking,
       raft.game_id,
       raft.round_id,
       raft.start_tick_id,
       raft.game_tick_rate,
       no.origin_id,
       cocc.num_clusters,
       coalesce(count(distinct tlacec.cover_edge_cluster_id),0) as looked_at_clusters_by_teammates
from team_looking_at_cover_edge_cluster tlacec
         join ticks t on tlacec.tick_id = t.id
         join rounds r on t.round_id = r.id
         join games g on r.game_id = g.id
         right join react_aim_and_fire_ticks raft on
            g.demo_file = raft.demo
            and raft.start_game_tick = t.game_tick_number
            and raft.spotter_id = tlacec.origin_player_id
         join nearest_origin no on raft.start_tick_id = no.tick_id and raft.spotter_id = no.player_id
         left join cover_origins_with_cluster_counts cocc on no.origin_id = cocc.index
-- need a bunch of extra cols in group by from raft for situations where tlacec is null
group by raft.demo, raft.visibility_technique_id, raft.spotter_id, raft.spotter, raft.spotted_id, raft.spotted,
         raft.start_game_tick, raft.end_game_tick, raft.next_start_game_tick, raft.last_end_game_tick,
         raft.react_aim_end_tick, raft.react_fire_end_tick, raft.distinct_others_spotted_during_time, raft.hacking,
         raft.game_id, raft.round_id, raft.start_tick_id, raft.game_tick_rate, no.origin_id, cocc.num_clusters
order by raft.demo, raft.start_game_tick;

select count(*) from react_aim_and_fire_ticks;
select count(*) from react_aim_fire_cover;
select player_id, tick_id, pos_x, pos_y, pos_z from react_aim_fire_cover r join player_at_tick pat on r.spotter_id = pat.player_id and r.start_tick_id = pat.tick_id where num_clusters is null and player_id = 123 and tick_id = 1554092;
select * from player_at_tick where player_id = 123 and tick_id = 1554092;
select * from cover_origins where x >= 380 and x <= 441 and y >= 1680 and y <= 1740;
select * from nearest_origin where player_id = 123 and tick_id = 1554092;
select * from player_at_tick where id = 1670181;
select * from players where id = 126;
select * from games where id = 11;
select * from ticks where id = 1554092;

select * from react_aim_fire_cover;

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
from react_aim_fire_cover raft
         join games g on g.demo_file = raft.demo
         join round_start_end_tick rset
              on g.id = rset.game_id
                  and int8range(raft.start_game_tick, raft.end_game_tick) <@
                      int8range(rset.min_game_tick, rset.max_game_tick)
;


select count(*) from react_final;

select count(*) from react_aim_and_fire_ticks raft
    join games g on g.demo_file = raft.demo;

select count(ce.index) as num_edges, count(distinct ce.cluster_id) as num_clusters, co.x as x, co.y as y, co.z as z
from cover_edges ce
    join cover_origins co on co.index = ce.origin_id
group by origin_id, co.x, co.y, co.z
order by num_edges desc