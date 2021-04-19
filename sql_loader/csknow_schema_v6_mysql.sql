CREATE TABLE `game_types` (
  `id` bigint PRIMARY KEY,
  `table_type` string
);

CREATE TABLE `games` (
  `id` bigint PRIMARY KEY,
  `demo_file` string,
  `demo_tick_rate` double,
  `game_tick_rate` double,
  `game_type` bigint
);

CREATE TABLE `players` (
  `id` bigint PRIMARY KEY,
  `game_id` bigint,
  `name` string,
  `steam_id` bigint
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
  `winner` smallint
);

CREATE TABLE `ticks` (
  `id` bigint PRIMARY KEY,
  `round_id` bigint,
  `game_time` bigint,
  `warmup` boolean,
  `bomb_carrier` bigint,
  `bomb_x` double,
  `bomb_y` double,
  `bomb_z` double
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
  `money` int
);

CREATE TABLE `spotted` (
  `id` bigint PRIMARY KEY,
  `tick_id` bigint,
  `spotted_player` bigint,
  `spotter_player` bigint,
  `is_spotted` boolean
);

CREATE TABLE `weapon_fire` (
  `id` bigint PRIMARY KEY,
  `tick_id` bigint,
  `shooter` bigint,
  `weapon` smallint
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
  `hit_group` bigint
);

CREATE TABLE `grenades` (
  `id` bigint PRIMARY KEY,
  `thrower` bigint,
  `grenade_type` smallint,
  `throw_tick` bigint,
  `active_tick` bigint,
  `expired_tick` bigint,
  `destroy_tick` bigint
);

CREATE TABLE `grenade_trajectories` (
  `id` bigint PRIMARY KEY,
  `grenade_id` bigint,
  `id_per_grenade` int,
  `pos_x` double,
  `pos_y` double,
  `pos_z` double
);

CREATE TABLE `player_flashed` (
  `id` bigint PRIMARY KEY,
  `tick_id` bigint,
  `grenade_id` bigint,
  `thrower` bigint,
  `victim` bigint
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
  `penetrated_objects` integer
);

CREATE TABLE `plants` (
  `id` bigint PRIMARY KEY,
  `start_tick` bigint,
  `end_tick` bigint,
  `planter` bigint,
  `succesful` boolean
);

CREATE TABLE `defusal` (
  `id` bigint PRIMARY KEY,
  `plant_id` bigint,
  `start_tick` bigint,
  `end_tick` bigint,
  `defuser` bigint,
  `succesful` boolean
);

CREATE TABLE `explosion` (
  `id` bigint PRIMARY KEY,
  `plant_id` bigint,
  `tick_id` bigint
);

ALTER TABLE `games` ADD FOREIGN KEY (`game_type`) REFERENCES `game_types` (`id`);

ALTER TABLE `players` ADD FOREIGN KEY (`game_id`) REFERENCES `games` (`id`);

ALTER TABLE `rounds` ADD FOREIGN KEY (`game_id`) REFERENCES `games` (`id`);

ALTER TABLE `ticks` ADD FOREIGN KEY (`round_id`) REFERENCES `rounds` (`id`);

ALTER TABLE `ticks` ADD FOREIGN KEY (`bomb_carrier`) REFERENCES `players` (`id`);

ALTER TABLE `player_at_tick` ADD FOREIGN KEY (`player_id`) REFERENCES `players` (`id`);

ALTER TABLE `player_at_tick` ADD FOREIGN KEY (`tick_id`) REFERENCES `ticks` (`id`);

ALTER TABLE `player_at_tick` ADD FOREIGN KEY (`active_weapon`) REFERENCES `equipment` (`id`);

ALTER TABLE `player_at_tick` ADD FOREIGN KEY (`primary_weapon`) REFERENCES `equipment` (`id`);

ALTER TABLE `player_at_tick` ADD FOREIGN KEY (`secondary_weapon`) REFERENCES `equipment` (`id`);

ALTER TABLE `spotted` ADD FOREIGN KEY (`tick_id`) REFERENCES `ticks` (`id`);

ALTER TABLE `spotted` ADD FOREIGN KEY (`spotted_player`) REFERENCES `players` (`id`);

ALTER TABLE `spotted` ADD FOREIGN KEY (`spotter_player`) REFERENCES `players` (`id`);

ALTER TABLE `weapon_fire` ADD FOREIGN KEY (`tick_id`) REFERENCES `ticks` (`id`);

ALTER TABLE `weapon_fire` ADD FOREIGN KEY (`shooter`) REFERENCES `players` (`id`);

ALTER TABLE `weapon_fire` ADD FOREIGN KEY (`weapon`) REFERENCES `equipment` (`id`);

ALTER TABLE `player_hurt` ADD FOREIGN KEY (`tick_id`) REFERENCES `ticks` (`id`);

ALTER TABLE `player_hurt` ADD FOREIGN KEY (`victim`) REFERENCES `players` (`id`);

ALTER TABLE `player_hurt` ADD FOREIGN KEY (`attacker`) REFERENCES `players` (`id`);

ALTER TABLE `player_hurt` ADD FOREIGN KEY (`weapon`) REFERENCES `equipment` (`id`);

ALTER TABLE `player_hurt` ADD FOREIGN KEY (`hit_group`) REFERENCES `hit_groups` (`id`);

ALTER TABLE `grenades` ADD FOREIGN KEY (`thrower`) REFERENCES `players` (`id`);

ALTER TABLE `grenades` ADD FOREIGN KEY (`grenade_type`) REFERENCES `equipment` (`id`);

ALTER TABLE `grenades` ADD FOREIGN KEY (`throw_tick`) REFERENCES `ticks` (`id`);

ALTER TABLE `grenades` ADD FOREIGN KEY (`active_tick`) REFERENCES `ticks` (`id`);

ALTER TABLE `grenades` ADD FOREIGN KEY (`expired_tick`) REFERENCES `ticks` (`id`);

ALTER TABLE `grenades` ADD FOREIGN KEY (`destroy_tick`) REFERENCES `ticks` (`id`);

ALTER TABLE `grenade_trajectories` ADD FOREIGN KEY (`grenade_id`) REFERENCES `grenades` (`id`);

ALTER TABLE `player_flashed` ADD FOREIGN KEY (`tick_id`) REFERENCES `ticks` (`id`);

ALTER TABLE `player_flashed` ADD FOREIGN KEY (`grenade_id`) REFERENCES `grenades` (`id`);

ALTER TABLE `player_flashed` ADD FOREIGN KEY (`thrower`) REFERENCES `players` (`id`);

ALTER TABLE `player_flashed` ADD FOREIGN KEY (`victim`) REFERENCES `players` (`id`);

ALTER TABLE `kills` ADD FOREIGN KEY (`tick_id`) REFERENCES `ticks` (`id`);

ALTER TABLE `kills` ADD FOREIGN KEY (`killer`) REFERENCES `players` (`id`);

ALTER TABLE `kills` ADD FOREIGN KEY (`victim`) REFERENCES `players` (`id`);

ALTER TABLE `kills` ADD FOREIGN KEY (`weapon`) REFERENCES `equipment` (`id`);

ALTER TABLE `kills` ADD FOREIGN KEY (`assister`) REFERENCES `players` (`id`);

ALTER TABLE `plants` ADD FOREIGN KEY (`start_tick`) REFERENCES `ticks` (`id`);

ALTER TABLE `plants` ADD FOREIGN KEY (`end_tick`) REFERENCES `ticks` (`id`);

ALTER TABLE `plants` ADD FOREIGN KEY (`planter`) REFERENCES `players` (`id`);

ALTER TABLE `defusal` ADD FOREIGN KEY (`plant_id`) REFERENCES `plants` (`id`);

ALTER TABLE `defusal` ADD FOREIGN KEY (`start_tick`) REFERENCES `ticks` (`id`);

ALTER TABLE `defusal` ADD FOREIGN KEY (`end_tick`) REFERENCES `ticks` (`id`);

ALTER TABLE `defusal` ADD FOREIGN KEY (`defuser`) REFERENCES `players` (`id`);

ALTER TABLE `explosion` ADD FOREIGN KEY (`plant_id`) REFERENCES `plants` (`id`);

ALTER TABLE `explosion` ADD FOREIGN KEY (`tick_id`) REFERENCES `ticks` (`id`);
