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

CREATE INDEX "game_id_key" ON "rounds" ("game_id");

CREATE INDEX "tick_time_index" ON "ticks" ("round_id", "demo_tick_number");

CREATE INDEX "tick_index" ON "player_at_tick" ("tick_id");

CREATE INDEX "player_tick_index" ON "player_at_tick" ("player_id", "tick_id");

CREATE INDEX "tick_player_index" ON "player_at_tick" USING gist ("pos_x", "pos_y", "pos_z");

CREATE INDEX "spotted_tick_index" ON "spotted" ("tick_id");

CREATE INDEX "spotted_index" ON "spotted" ("spotted_player");

CREATE INDEX "spotter_index" ON "spotted" ("spotter_player");

CREATE INDEX "fire_tick_index" ON "weapon_fire" ("tick_id");

CREATE INDEX "fire_shooter_index" ON "weapon_fire" ("shooter");

CREATE INDEX "fire_weapon_index" ON "weapon_fire" ("weapon");

CREATE INDEX "hurt_tick_index" ON "hurt" ("tick_id");

CREATE INDEX "hurt_victim_index" ON "hurt" ("victim");

CREATE INDEX "hurt_attacker_index" ON "hurt" ("attacker");

CREATE INDEX "hurt_weapon_index" ON "hurt" ("weapon");

CREATE INDEX "thrower_index" ON "grenades" ("thrower");

CREATE INDEX "throw_index" ON "grenades" ("throw_tick");

CREATE INDEX "active_index" ON "grenades" ("active_tick");

CREATE INDEX "expired_index" ON "grenades" ("expired_tick");

CREATE INDEX "destroyed_index" ON "grenades" ("destroy_tick");

CREATE INDEX "trajectory_index" ON "grenade_trajectories" ("grenade_id");

CREATE INDEX "flashed_grenade_index" ON "flashed" ("grenade_id");

CREATE INDEX "flashed_tick_index" ON "flashed" ("tick_id");

CREATE INDEX "flashed_thrower_index" ON "flashed" ("thrower");

CREATE INDEX "flashed_victim_index" ON "flashed" ("victim");

CREATE INDEX "kills_tick_index" ON "kills" ("tick_id");

CREATE INDEX "kills_killer_index" ON "kills" ("killer");

CREATE INDEX "kills_victim_index" ON "kills" ("victim");

CREATE INDEX "kills_assister_index" ON "kills" ("assister");

CREATE INDEX "kills_weapon_index" ON "kills" ("weapon");

CREATE INDEX "plant_planter_index" ON "plants" ("planter");

CREATE INDEX "plant_start_index" ON "plants" ("start_tick");

CREATE INDEX "plant_end_index" ON "plants" ("end_tick");

CREATE INDEX "defusals_id_index" ON "defusals" ("plant_id");

CREATE INDEX "defusals_defuser_index" ON "defusals" ("defuser");

CREATE INDEX "defusals_start_index" ON "defusals" ("start_tick");

CREATE INDEX "defusals_end_index" ON "defusals" ("end_tick");

CREATE INDEX "explosions_plant_index" ON "explosions" ("plant_id");

CREATE INDEX "explosions_tick_index" ON "explosions" ("tick_id");
