ALTER TABLE "games" ADD FOREIGN KEY ("game_type") REFERENCES "game_types" ("id");

ALTER TABLE "players" ADD FOREIGN KEY ("game_id") REFERENCES "games" ("id");

ALTER TABLE "rounds" ADD FOREIGN KEY ("game_id") REFERENCES "games" ("id");

ALTER TABLE "ticks" ADD FOREIGN KEY ("round_id") REFERENCES "rounds" ("id");

ALTER TABLE "ticks" ADD FOREIGN KEY ("bomb_carrier") REFERENCES "players" ("id");

ALTER TABLE "player_at_tick" ADD FOREIGN KEY ("player_id") REFERENCES "players" ("id");

ALTER TABLE "player_at_tick" ADD FOREIGN KEY ("tick_id") REFERENCES "ticks" ("id");

ALTER TABLE "player_at_tick" ADD FOREIGN KEY ("active_weapon") REFERENCES "equipment" ("id");

ALTER TABLE "player_at_tick" ADD FOREIGN KEY ("primary_weapon") REFERENCES "equipment" ("id");

ALTER TABLE "player_at_tick" ADD FOREIGN KEY ("secondary_weapon") REFERENCES "equipment" ("id");

ALTER TABLE "spotted" ADD FOREIGN KEY ("tick_id") REFERENCES "ticks" ("id");

ALTER TABLE "spotted" ADD FOREIGN KEY ("spotted_player") REFERENCES "players" ("id");

ALTER TABLE "spotted" ADD FOREIGN KEY ("spotter_player") REFERENCES "players" ("id");

ALTER TABLE "weapon_fire" ADD FOREIGN KEY ("tick_id") REFERENCES "ticks" ("id");

ALTER TABLE "weapon_fire" ADD FOREIGN KEY ("shooter") REFERENCES "players" ("id");

ALTER TABLE "weapon_fire" ADD FOREIGN KEY ("weapon") REFERENCES "equipment" ("id");

ALTER TABLE "hurt" ADD FOREIGN KEY ("tick_id") REFERENCES "ticks" ("id");

ALTER TABLE "hurt" ADD FOREIGN KEY ("victim") REFERENCES "players" ("id");

ALTER TABLE "hurt" ADD FOREIGN KEY ("attacker") REFERENCES "players" ("id");

ALTER TABLE "hurt" ADD FOREIGN KEY ("weapon") REFERENCES "equipment" ("id");

ALTER TABLE "hurt" ADD FOREIGN KEY ("hit_group") REFERENCES "hit_groups" ("id");

ALTER TABLE "grenades" ADD FOREIGN KEY ("thrower") REFERENCES "players" ("id");

ALTER TABLE "grenades" ADD FOREIGN KEY ("grenade_type") REFERENCES "equipment" ("id");

ALTER TABLE "grenades" ADD FOREIGN KEY ("throw_tick") REFERENCES "ticks" ("id");

ALTER TABLE "grenades" ADD FOREIGN KEY ("active_tick") REFERENCES "ticks" ("id");

ALTER TABLE "grenades" ADD FOREIGN KEY ("expired_tick") REFERENCES "ticks" ("id");

ALTER TABLE "grenades" ADD FOREIGN KEY ("destroy_tick") REFERENCES "ticks" ("id");

ALTER TABLE "grenade_trajectories" ADD FOREIGN KEY ("grenade_id") REFERENCES "grenades" ("id");

ALTER TABLE "flashed" ADD FOREIGN KEY ("tick_id") REFERENCES "ticks" ("id");

ALTER TABLE "flashed" ADD FOREIGN KEY ("grenade_id") REFERENCES "grenades" ("id");

ALTER TABLE "flashed" ADD FOREIGN KEY ("thrower") REFERENCES "players" ("id");

ALTER TABLE "flashed" ADD FOREIGN KEY ("victim") REFERENCES "players" ("id");

ALTER TABLE "kills" ADD FOREIGN KEY ("tick_id") REFERENCES "ticks" ("id");

ALTER TABLE "kills" ADD FOREIGN KEY ("killer") REFERENCES "players" ("id");

ALTER TABLE "kills" ADD FOREIGN KEY ("victim") REFERENCES "players" ("id");

ALTER TABLE "kills" ADD FOREIGN KEY ("weapon") REFERENCES "equipment" ("id");

ALTER TABLE "kills" ADD FOREIGN KEY ("assister") REFERENCES "players" ("id");

ALTER TABLE "plants" ADD FOREIGN KEY ("start_tick") REFERENCES "ticks" ("id");

ALTER TABLE "plants" ADD FOREIGN KEY ("end_tick") REFERENCES "ticks" ("id");

ALTER TABLE "plants" ADD FOREIGN KEY ("planter") REFERENCES "players" ("id");

ALTER TABLE "defusals" ADD FOREIGN KEY ("plant_id") REFERENCES "plants" ("id");

ALTER TABLE "defusals" ADD FOREIGN KEY ("start_tick") REFERENCES "ticks" ("id");

ALTER TABLE "defusals" ADD FOREIGN KEY ("end_tick") REFERENCES "ticks" ("id");

ALTER TABLE "defusals" ADD FOREIGN KEY ("defuser") REFERENCES "players" ("id");

ALTER TABLE "explosions" ADD FOREIGN KEY ("plant_id") REFERENCES "plants" ("id");

ALTER TABLE "explosions" ADD FOREIGN KEY ("tick_id") REFERENCES "ticks" ("id");
