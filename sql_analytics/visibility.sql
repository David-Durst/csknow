drop table visibilities;
create temp table visibilities as
select h.index, h.demo, min(t.id) as tick_id, h.spotter_id, h.spotted_id, h.spotted, h.start_game_tick, h.end_game_tick, h.next_start_game_tick, min(t.game_tick_number) as automated_vis_tick
from spotted s
         join ticks t on s.tick_id = t.id
         right join hand_visibility h
                    on h.spotted_id = s.spotted_player
                        and h.spotter_id = s.spotter_player
                        and h.start_game_tick <= t.game_tick_number
                        and h.next_start_game_tick >= t.game_tick_number
                        and s.is_spotted = true
group by h.index, h.demo, h.spotter_id, h.spotted_id, h.spotted, h.start_game_tick, h.end_game_tick, h.next_start_game_tick
order by h.index;

drop table react_ticks;
create temp table react_ticks as
select v.index, v.demo, v.tick_id, v.spotter_id, v.spotted_id, v.spotted, v.start_game_tick, v.end_game_tick, v.next_start_game_tick, v.automated_vis_tick, min(t.game_tick_number) as react_end_tick
from lookers l
         join ticks t on l.tick_id = t.id
         right join visibilities v
                    on v.start_game_tick <= t.game_tick_number
                        and v.next_start_game_tick >= t.game_tick_number
                        and l.looker_player_id = v.spotter_id
                        and l.looked_at_player_id = v.spotted_id
group by v.index, v.demo, v.tick_id, v.spotter_id, v.spotted_id, v.spotted, v.start_game_tick, v.end_game_tick, v.next_start_game_tick, v.automated_vis_tick
order by v.index;

select *, (react_end_tick - start_game_tick) / 64.0 as hand_react_ticks, (react_end_tick - react_ticks.automated_vis_tick) / 64.0 as automated_react_ticks
from react_ticks;

select *
from spotted s
join ticks t on s.tick_id = t.id
join rounds r on t.round_id = r.id
join games g on r.game_id = g.id
where game_tick_number >= 7150 and game_tick_number <= 8000
    and g.id = 0
    and s.spotted_player = 1 and s.spotter_player = 2
order by t.id;
