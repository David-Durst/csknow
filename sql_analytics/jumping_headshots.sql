select *
from ticks t
         join hurt h on t.id = h.tick_id
         join player_at_tick pat on h.tick_id = pat.tick_id and attacker = player_id
where is_airborne and hit_group = 1
limit 10;
# note: h.weapon may not equal pat.active_weapon if headshotting attacker dies on same tick, below queries show that
# for round_id 723, attacker 238, game_tick_number 169063
select * from hit_groups;
select * from equipment;
select * from kills join ticks t on kills.tick_id = t.id where t.round_id = 723;