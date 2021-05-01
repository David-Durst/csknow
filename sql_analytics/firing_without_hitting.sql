drop table firing_no_hits;
create temporary table firing_no_hits (
    select *, game_tick_number - last_hit_neg as ticks_between_hits
    from (
             select *,
                    sum(case
                            when victim is not null then game_tick_number
                            when num_hits = 0 then -1
                            else 0 end)
                        over (partition by round_id, num_hits order by tick_id) as last_hit_neg
             from (
                      select g.id                                           as game_id,
                             r.id                                           as round_id,
                             t.id                                           as tick_id,
                             t.game_tick_number,
                             wf.shooter,
                             e.name                                         as weapon_name,
                             h.victim,
                             start_ticks.game_tick_number                   as start_game_tick_number,
                             sum(case when h.victim is not null then 1 else 0 end)
                                 over (partition by round_id order by t.id, shooter, victim) as num_hits
                      from ticks t
                               join rounds r on t.round_id = r.id
                               join games g on r.game_id = g.id
                               inner join weapon_fire wf on t.id = wf.tick_id
                               join equipment e on e.id = wf.weapon
                               left join hurt h on t.id = h.tick_id
                               join ticks start_ticks on start_ticks.id = r.start_tick
                  ) i1
         ) i2
    where last_hit_neg > 0
);

# this finds longest times between hits, not tight sequences of misses - the one in round 166 is a save, where just knifing at end without point
select ticks_between_hits, round_id, game_tick_number
from (
     select ticks_between_hits, round_id, game_tick_number
     from (
              select ticks_between_hits,
                     last_hit_neg,
                     lead(last_hit_neg) over (order by tick_id, shooter, victim) as next_last_hit,
                     round_id,
                     game_tick_number
              from firing_no_hits
              order by tick_id, shooter, victim
          ) i1
     where next_last_hit != last_hit_neg and ticks_between_hits >= 0
     ) i2
order by ticks_between_hits DESC
limit 10000;


select *,
       sum(case when game_tick_number - last_game_tick < 128 then 0 else 1 end)
           over (partition by round_id order by tick_id) as burst_id
from (
         select ticks_between_hits,
                last_hit_neg,
                lag(game_tick_number) over (order by tick_id, shooter, victim) as last_game_tick,
                round_id,
                game_tick_number,
                tick_id
         from firing_no_hits
         order by tick_id, shooter, victim
     ) i1
