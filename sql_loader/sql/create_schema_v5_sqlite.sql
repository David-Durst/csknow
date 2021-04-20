CREATE TABLE `game_types` (
  `id` bigint PRIMARY KEY,
  `table_type` string
);

CREATE TABLE `games` (
  `id` bigint PRIMARY KEY,
  `demo_file` string,
  `demo_tick_rate` double,
  `game_tick_rate` double,
  `game_type` bigint,
  FOREIGN KEY(`game_type`) REFERENCES game_types(id)
);

CREATE TABLE `players` (
  `id` bigint PRIMARY KEY,
  `game_id` bigint,
  `name` string,
  `steam_id` bigint,
  FOREIGN KEY(`game_id`) REFERENCES games(id)
);

CREATE TABLE `equipment` (
  `id` smallint PRIMARY KEY,
  `name` string
);

CREATE TABLE `rounds` (
  `id` bigint PRIMARY KEY,
  `game_id` bigint,
  `start_tick` bigint,
  `end_tick` bigint,
  `freeze_time_end` bigint,
  `round_number` smallint,
  `round_end_reason` smallint,
  `winner` smallint,
  FOREIGN KEY(`game_id`) REFERENCES games(id)
);

CREATE TABLE `ticks` (
  `id` bigint PRIMARY KEY,
  `round_id` bigint,
  `game_time` bigint,
  `warmup` boolean,
  `bomb_carrier` bigint,
  `bomb_x` double,
  `bomb_y` double,
  `bomb_z` double,
  FOREIGN KEY(`round_id`) REFERENCES rounds(id)
);

CREATE TABLE `player_at_tick` (
  `id` bigint PRIMARY KEY,
  `player_id` bigint,
  `tick_id` bigint,
  `pos_x` double,
  `pos_y` double,
  `pos_z` double,
  `view_x` double,
  `view_y` double,
  `health` double,
  `armor` double,
  `has_helmet` boolean,
  `is_alive` boolean,
  `is_crouching` boolean,
  `is_airborne` boolean,
  `remaining_flash_time` double,
  `active_weapon` smallint,
  `primary_weapon` smallint,
  `primary_bullets_clip` smallint,
  `primary_bullets_reserve` smallint,
  `secondary_weapon` smallint,
  `secondary_bullets_clip` smallint,
  `secondary_bullets_reserve` smallint,
  `num_he` smallint,
  `num_flash` smallint,
  `num_smoke` smallint,
  `num_molotov` smallint,
  `num_incendiary` smallint,
  `num_decoy` smallint,
  `num_zeus` smallint,
  `has_defuser` boolean,
  `has_bomb` boolean,
  `money` int,
  FOREIGN KEY(`player_id`) REFERENCES players(id),
  FOREIGN KEY(`tick_id`) REFERENCES ticks(id),
  FOREIGN KEY(`active_weapon`) REFERENCES equipment(id),
  FOREIGN KEY(`primary_weapon`) REFERENCES equipment(id),
  FOREIGN KEY(`secondary_weapon`) REFERENCES equipment(id)
);

CREATE TABLE `spotted` (
  `id` bigint PRIMARY KEY,
  `tick_id` bigint,
  `spotted_player` bigint,
  `spotter_player` bigint,
  `is_spotted` boolean,
  FOREIGN KEY(`tick_id`) REFERENCES ticks(id),
  FOREIGN KEY(`spotted_player`) REFERENCES players(id),
  FOREIGN KEY(`spotter_player`) REFERENCES players(id)
);

CREATE TABLE `weapon_fire` (
  `id` bigint PRIMARY KEY,
  `tick_id` bigint,
  `shooter` bigint,
  `weapon` smallint,
  FOREIGN KEY(`tick_id`) REFERENCES ticks(id),
  FOREIGN KEY(`shooter`) REFERENCES players(id),
  FOREIGN KEY(`weapon`) REFERENCES equipment(id)
);

CREATE TABLE `hit_groups` (
  `id` bigint PRIMARY KEY,
  `group_name` string
);

CREATE TABLE `player_hurt` (
  `id` bigint PRIMARY KEY,
  `tick_id` bigint,
  `victim` bigint,
  `attacker` bigint,
  `weapon` smallint,
  `armor_damage` integer,
  `armor` integer,
  `health_damage` integer,
  `health` integer,
  `hit_group` bigint,
  FOREIGN KEY(`tick_id`) REFERENCES ticks(id),
  FOREIGN KEY(`victim`) REFERENCES players(id),
  FOREIGN KEY(`attacker`) REFERENCES players(id),
  FOREIGN KEY(`weapon`) REFERENCES equipment(id),
  FOREIGN KEY(`hit_group`) REFERENCES hit_groups(id)
);

CREATE TABLE `grenades` (
  `id` bigint PRIMARY KEY,
  `thrower` bigint,
  `grenade_type` smallint,
  `throw_tick` bigint,
  `active_tick` bigint,
  `expired_tick` bigint,
  `destroy_tick` bigint,
  FOREIGN KEY(`grenade_type`) REFERENCES equipment(id),
  FOREIGN KEY(`throw_tick`) REFERENCES ticks(id),
  FOREIGN KEY(`active_tick`) REFERENCES ticks(id),
  FOREIGN KEY(`expired_tick`) REFERENCES ticks(id),
  FOREIGN KEY(`destroy_tick`) REFERENCES ticks(id)
);

CREATE TABLE `grenade_trajectories` (
  `id` bigint PRIMARY KEY,
  `grenade_id` bigint,
  `id_per_grenade` int,
  `pos_x` double,
  `pos_y` double,
  `pos_z` double,
  FOREIGN KEY(`grenade_id`) REFERENCES grenades(id)
);

CREATE TABLE `player_flashed` (
  `id` bigint PRIMARY KEY,
  `tick_id` bigint,
  `grenade_id` bigint,
  `thrower` bigint,
  `victim` bigint,
  FOREIGN KEY(`tick_id`) REFERENCES ticks(id),
  FOREIGN KEY(`grenade_id`) REFERENCES grenades(id),
  FOREIGN KEY(`thrower`) REFERENCES players(id),
  FOREIGN KEY(`victim`) REFERENCES players(id)
);

CREATE TABLE `kills` (
  `id` bigint PRIMARY KEY,
  `tick_id` bigint,
  `killer` bigint,
  `victim` bigint,
  `weapon` smallint,
  `assister` bigint,
  `is_headshot` boolean,
  `is_wallbang` boolean,
  `penetrated_objects` integer,
  FOREIGN KEY(`tick_id`) REFERENCES ticks(id),
  FOREIGN KEY(`killer`) REFERENCES players(id),
  FOREIGN KEY(`victim`) REFERENCES players(id)
  FOREIGN KEY(`weapon`) REFERENCES equipment(id)
  FOREIGN KEY(`assister`) REFERENCES players(id)
);

CREATE TABLE `plants` (
  `id` bigint PRIMARY KEY,
  `start_tick` bigint,
  `end_tick` bigint,
  `planter` bigint,
  `succesful` boolean,
  FOREIGN KEY(`start_tick`) REFERENCES ticks(id),
  FOREIGN KEY(`end_tick`) REFERENCES ticks(id),
  FOREIGN KEY(`planter`) REFERENCES players(id)
);

CREATE TABLE `defusal` (
  `id` bigint PRIMARY KEY,
  `plant_id` bigint,
  `start_tick` bigint,
  `end_tick` bigint,
  `defuser` bigint,
  `succesful` boolean,
  FOREIGN KEY(`plant_id`) REFERENCES plants(id),
  FOREIGN KEY(`start_tick`) REFERENCES ticks(id),
  FOREIGN KEY(`end_tick`) REFERENCES ticks(id),
  FOREIGN KEY(`defuser`) REFERENCES players(id)
);

CREATE TABLE `explosion` (
  `id` bigint PRIMARY KEY,
  `plant_id` bigint,
  `tick_id` bigint,
  FOREIGN KEY(`plant_id`) REFERENCES plants(id),
  FOREIGN KEY(`tick_id`) REFERENCES ticks(id)
);
