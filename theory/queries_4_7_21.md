# Introduction
This document lists the current set of supported or in development queries for CSKnow as of 4-7-21.
For each query, this document lists:

1. Summary
2. English definition
3. In-Game Examples
4. Pseudocode definition

The pseudocode definitions assume knowledge of the schema specified in 
[schema.md](https://github.com/David-Durst/csknow/blob/master/theory/schema.md).

# Queries

## Wallers

### Summary
Situations where it appears that a player can see enemies through walls.

### English Definition
1. Game state specifies that player A can't see player B
2. Ray from A's crosshair intersects B's bounding box for two seconds

### In-Game Examples
[![](http://img.youtube.com/vi/o_4wTBSnopA/0.jpg)](http://www.youtube.com/watch?v=o_4wTBSnopA "Video Example of a Waller")

The above is a video example of a waller. It shows the same situations as the 2D
visualization below. Player A (BOT Moe) is running through lower tunnels of 
the de_dust2 map and staring at player B (BOT Vitaly) through the wall.

![Wallers example image](example_images/wallers.png)

### Pseudocode Definition
1. For every tick `j`
2. For all pairs of players `(A,B)` on different teams
3. If `STARING_AT(A,B,k)` and `MOST_RECENT_ENTRY(SPOTTED,A,B,k)` for all `j < k < j+TWO_SECONDS`
   1. `STARING_AT` - computed from `playerPosition` table - for `de`
4. Then `WALLER(A,B,j)`.

## Baiters

### Summary
Situations where players could help their teammates by moving to their position,
but instead hide and let their teammates die.

### English Definition
1. 



