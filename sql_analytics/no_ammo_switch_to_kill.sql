select *
from ticks t
    join player_at_tick pat on t.id = pat.tick_id
    join kills k on pat.tick_id >= k.tick_id #and (t.id >= k.tick_id)
    #join rounds r on t.round_id = r.id
    #join games g on r.game_id = g.id
#where pat.active_weapon != k.weapon and (
#        (pat.primary_weapon = pat.active_weapon and pat.primary_bullets_clip = 0) or
#        (pat.primary_weapon = pat.secondary_weapon and pat.secondary_bullets_clip = 0)
#    )
limit 10

set infinidb_vtable_mode = 0;