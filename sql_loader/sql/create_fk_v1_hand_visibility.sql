ALTER TABLE "nearest_origin" ADD FOREIGN KEY ("tick_id") REFERENCES "ticks" ("id");

ALTER TABLE "nearest_origin" ADD FOREIGN KEY ("pat_id") REFERENCES "player_at_tick" ("id");

ALTER TABLE "team_looking_at_cover_edge_cluster" ADD FOREIGN KEY ("tick_id") REFERENCES "ticks" ("id");
