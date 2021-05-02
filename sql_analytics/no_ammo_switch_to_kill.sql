# nothing for 1 second, but 2 seconds do have results
select t.id, max(pat.tick_id), max(pat.active_weapon),
       max(pat.primary_weapon), max(pat.primary_bullets_clip),
       max(pat.secondary_weapon), max(pat.secondary_bullets_clip),
       max(k.weapon), max(k.killer)
from kills k
     join ticks t on k.tick_id = t.id
     join rounds r on t.round_id = r.id
     join games g on r.game_id = g.id
     join player_at_tick pat on
     k.tick_id >= pat.tick_id and k.tick_id <= 2 * g.demo_tick_rate + pat.tick_id and k.killer = pat.player_id
where pat.active_weapon != k.weapon and (
        (pat.primary_weapon = pat.active_weapon and pat.primary_bullets_clip = 0) or
        (pat.primary_weapon = pat.secondary_weapon and pat.secondary_bullets_clip = 0)
    ) and pat.active_weapon > 0 and k.weapon > 0 and pat.active_weapon < 400 and k.weapon < 400
group by t.id
limit 10;
