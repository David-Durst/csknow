drop table multiple_attacks;
# need temporary table hack because mariadb claims it's a circular join to use the pat.player_id as a join predicate for i1
create temporary table multiple_attacks (
    select t.id,
           round_id,
           i1.tick_id    as hurt_tick_id,
           victim,
           attacker1,
           weapon1,
           attacker2,
           weapon2,
           pat.player_id as other_pid
    from ticks t
             join player_at_tick pat on t.id = pat.tick_id
             join rounds r on t.round_id = r.id
             join
         (
             select h1.tick_id  as tick_id,
                    h1.victim   as victim,
                    h2.victim   as _,
                    h1.attacker as attacker1,
                    h2.attacker as attacker2,
                    h1.weapon   as weapon1,
                    h2.weapon   as weapon2,
                    r2.id       as double_hurt_round_id
             from hurt h1
                      join hurt h2
                           on h1.victim = h2.victim and h1.attacker < h2.attacker and h1.tick_id = h2.tick_id and
                              h1.weapon = h2.weapon
                      join ticks t2 on h1.tick_id = t2.id
                      join rounds r2 on t2.round_id = r2.id
         ) i1 on r.id = double_hurt_round_id
    where t.id = r.end_tick
      and pat.is_alive
);
select * from multiple_attacks where other_pid = victim


