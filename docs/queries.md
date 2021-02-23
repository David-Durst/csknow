* Queries
These queries are important both from a game perspective (answer frustrations of mine) and a technical one (lots of intersection computations or spatio-temporal joins),
1. Find the wallers. Compute the players (across multiple games) whose cursor points at people they can't see most frequently.
    a. context - wallers are a type of cheater who can see people through walls.
    b. Type of query - Sort of group by of spatio-temporal events
    c. Optimal Implementation - spatio-temporal index on tick table for each game to filter for vector-rectangle intersections, then aggregate.
1. How bad is the netcode? Compute all the times where players fire, they are aiming at someone, standing still, 
    using an accurate weapon (AK,M4,AWP,Scout,SG,AUG), but don't hit. 
    a. context - everyone playing CSGO feels like they get robbed all the time by bad netcode not counting hits. Let's see if this happerns
    b. Type of query - Aggregation of spatio-temporal events
    c. Optimal Implementation - Join shots with tick table, then for each shot check if vector-rectangle intersection works for each player, then aggregate.
1. How do people peek with different AK vs other weapons? For 2-3 corners of interest, for each time a player peeks those corners,
    find the most similar trajectory with another weapon leading up to getting to that corner, and see how the trajectories diverge
    a. context - Do people play differently with different weapons? Let's understand how weapon causes game decisions
    b. Type of query - spatio-temporal join
    c. Optimal Implementation - filter by trajectories moving towards region of map, then join based on trajectory similarity

** Simpler Queries
These will be less taxing on the system. I'll come back to them later.
1. Compute all the times players got shot before they could see the enemy (i.e. they were shot and the enemy could only see them for less than 100ms).
   a. context - Players see enemies 100ms in the past. If an attacker runs around the corner, they will see the defender
   immediately (as the defender is standing still) but the defender only sees the attacker 100ms later. This "peekers advantage"
   is one of the biggest frustrations in FPS.
   b. Type of query - aggregation and filter of temporal sequence of events
   c. Optimal Implementation - join deaths table with visibility table
1. Which weapons beat other weapons categorized by: (1) average velocity 0.5s before kill, (2) distance of kill, and (3)
   whether weapon is used semi-auto/full-auto/bursting
   a. context - Game designers balance weapons to be used in certain situations. Let's see if they are actually used
   as intended.
   b. Type of query - Group by of temporal sequence of events
   c. Optimal Implementation - join deaths table with visibility table

** Technical Details
1. Could replace vector-rectangle details with more complicated approximations of hit detection (https://www.hltv.org/news/8335/csgo-hitboxes-explained)
but will come back to this later.