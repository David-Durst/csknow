select *
from (
     select *,
            sum(abs(last_view_delta)) over (partition by round_id, player_id, start_of_turn order by tick_id) as turn_amount
     from (
              select *,
                     sum(case when sign(last_view_delta) != sign(prior_view_delta) or last_view_delta > 100 then 1 else 0 end)
                         over (partition by round_id, player_id order by tick_id) as start_of_turn
              from (
                       select *,
                              case
                                  when view_x > 350 and last_view_x < 10 then -1 * (last_view_x + 360 - view_x)
                                  when last_view_x > 350 and view_x < 10 then view_x + 360 - last_view_x
                                  else view_x - last_view_x end
                                  as last_view_delta,
                              case
                                  when last_view_x > 350 and prior_view_x < 10 then -1 * (prior_view_x + 360 - last_view_x)
                                  when prior_view_x > 350 and last_view_x < 10 then last_view_x + 360 - prior_view_x
                                  else last_view_x - prior_view_x end
                                  as prior_view_delta
                       from (
                                select pat.tick_id                                      as tick_id,
                                       pat.player_id,
                                       pat.view_x,
                                       g.game_tick_rate,
                                       t.demo_tick_number,
                                       t.game_tick_number,
                                       pat.is_airborne,
                                       warmup,
                                       round_id,
                                       attacker,
                                       lag(pat.view_x) over (partition by round_id, player_id order by tick_id) as last_view_x,
                                       lag(pat.view_x,2) over (partition by round_id, player_id order by tick_id) as prior_view_x
                                from ticks t
                                        join player_at_tick pat on t.id = pat.tick_id
                                        left join hurt h on pat.tick_id = h.tick_id and pat.player_id = h.attacker
                                        join rounds r on t.round_id = r.id
                                        join games g on r.game_id = g.id
                                where game_id < 2
                            ) i1
                   ) i2
          ) i3
     ) i4
where turn_amount > 310 and attacker is not null
limit 100;
