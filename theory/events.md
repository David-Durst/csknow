1. Each event defines a relationship
1. Relationships can be
    1. predicate 
    1. one-to-one
    2. one-to-many
    3. many-to-many
1. Events may describe relationship updates that are
    1. state dependent - requires checking events for an undefined window around current tick
    2. state independent - requires checking events for a fixed window around current tick
1. events 
    1. tick
    2. spotted change - either a player is now visible or now not visible by another player
    3. weapon fire
    4. player hurt - this doesn't have to be from a weapon. There is also damage from world and c4.
    5. grenade thrown
    6. player killed
1. relationships described by events
    1. is player in location (cat, long, mid) - predicate, state independent
    2. is player current visible to other player - one-to-one, state dependent
    3. is player firing during the current second (32 ticks) - predicate, state independent
    4. is player hurt during the current second (32 ticks) - predicate, state independent
    5. is player throwing a grenade during the current second (32 ticks) - 