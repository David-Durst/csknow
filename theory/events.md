1. Each data point collected from the CSGO game engine in the demo file is an event.
    1. **demo file** - a log of all the events that occured during the game. 
       It can be used to replay the game in the game engine.
    1. **event** - a tick (a moment in time) when the game engine logged that something happened.
    2. **tick** - the discrete unit of time used by the engine.
    3. **tick rate** is the number of ticks per second
    4. There are many kinds of tick rates including
        1. **game tick rate** - the tick rate the game engine uses for drawing updates to the screen
        2. **physics engine tick rate** - in some games (not CSGO, but Valorant), 
           the game engine uses a separate tick rate for computing physics of objects
        3. **demo tick rate** - the tickr rate the game engine uses to log events to the demo file.
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
    5. is player throwing a grenade during the current second (32 ticks) - predicate, state independent
    6. is more players moving and within 100 units of at least 3 other players 
       over a window of at least 96 ticks - one-to-many, state dependent