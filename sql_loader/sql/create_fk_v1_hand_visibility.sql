ALTER TABLE "nearest_origin" ADD FOREIGN KEY ("tick_id") REFERENCES "ticks" ("id");

ALTER TABLE "nearest_origin" ADD FOREIGN KEY ("pat_id") REFERENCES "player_at_tick" ("id");

ALTER TABLE "team_looking_at_cover_edge_cluster" ADD FOREIGN KEY ("tick_id") REFERENCES "ticks" ("id");

CREATE MATERIALIZED VIEW ticks_rounds_games AS 
    select 
        t.id as tick_id, 
	t.round_id as round_id,
	t.demo_tick_number as demo_tick_number,
	t.game_tick_number as game_tick_number,
	lead(t.game_tick_number) over (partition by g.id order by t.id) as next_game_tick_number,
	int8range(game_tick_number, lead(t.game_tick_number) over (partition by g.id order by t.id), '[)') as game_tick_range,
	t.bomb_carrier as bomb_carrier,
	t.bomb_x as bomb_x,
	t.bomb_y as bomb_y,
	t.bomb_z as bomb_z,
	r.game_id, 
	g.demo_file, 
	g.game_tick_rate, 
	g.demo_tick_rate
    from ticks t
        join rounds r on t.round_id = r.id
        join games g on r.game_id = g.id
    order by t.id
;

CREATE UNIQUE INDEX ticks_rounds_games_tick_id on ticks_rounds_games (tick_id);

CREATE UNIQUE INDEX ticks_rounds_games_tick_game_tick_number on ticks_rounds_games (demo_file, game_tick_number, next_game_tick_number);
CREATE INDEX ticks_rounds_games_tick_game_tick_range on ticks_rounds_games using gist (game_tick_range);


