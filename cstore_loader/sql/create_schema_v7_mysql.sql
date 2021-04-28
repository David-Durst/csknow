CREATE TABLE `game_types` (
  `id` bigint,
  `table_type` varchar(20)
) engine=columnstore;

CREATE TABLE `games` (
  `id` bigint,
  `demo_file` varchar(1000),
  `demo_tick_rate` double,
  `game_tick_rate` double,
  `game_type` bigint
) engine=columnstore;

CREATE TABLE `players` (
  `id` bigint,
  `game_id` bigint,
  `name` varchar(255),
  `steam_id` bigint
) engine=columnstore;

CREATE TABLE `equipment` (
  `id` smallint,
  `name` varchar(30)
) engine=columnstore;

CREATE TABLE `rounds` (
  `id` bigint,
  `game_id` bigint,
  `start_tick` bigint,
  `end_tick` bigint,
  `warmup` boolean,
  `freeze_time_end` bigint,
  `round_number` smallint,
  `round_end_reason` smallint,
  `winner` smallint,
  `t_wins` smallint,
  `ct_wins` smallint
) engine=columnstore;

CREATE TABLE `ticks` (
  `id` bigint,
  `round_id` bigint,
  `game_time` bigint,
  `demo_tick_number` bigint,
  `game_tick_number` bigint,
  `bomb_carrier` bigint,
  `bomb_x` double,
  `bomb_y` double,
  `bomb_z` double
) engine=columnstore;

CREATE TABLE `player_at_tick` (
  `id` bigint,
  `player_id` bigint,
  `tick_id` bigint,
  `pos_x` double,
  `pos_y` double,
  `pos_z` double,
  `view_x` double,
  `view_y` double,
  `team` smallint,
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
) engine=columnstore;

CREATE TABLE `spotted` (
  `id` bigint,
  `tick_id` bigint,
  `spotted_player` bigint,
  `spotter_player` bigint,
  `is_spotted` boolean
) engine=columnstore;

CREATE TABLE `weapon_fire` (
  `id` bigint,
  `tick_id` bigint,
  `shooter` bigint,
  `weapon` smallint
) engine=columnstore;

CREATE TABLE `hit_groups` (
  `id` bigint,
  `group_name` varchar(30)
) engine=columnstore;

CREATE TABLE `hurt` (
  `id` bigint,
  `tick_id` bigint,
  `victim` bigint,
  `attacker` bigint,
  `weapon` smallint,
  `armor_damage` integer,
  `armor` integer,
  `health_damage` integer,
  `health` integer,
  `hit_group` bigint
) engine=columnstore;

CREATE TABLE `grenades` (
  `id` bigint,
  `thrower` bigint,
  `grenade_type` smallint,
  `throw_tick` bigint,
  `active_tick` bigint,
  `expired_tick` bigint,
  `destroy_tick` bigint
) engine=columnstore;

CREATE TABLE `grenade_trajectories` (
  `id` bigint,
  `grenade_id` bigint,
  `id_per_grenade` int,
  `pos_x` double,
  `pos_y` double,
  `pos_z` double
) engine=columnstore;

CREATE TABLE `flashed` (
  `id` bigint,
  `tick_id` bigint,
  `grenade_id` bigint,
  `thrower` bigint,
  `victim` bigint
) engine=columnstore;

CREATE TABLE `kills` (
  `id` bigint,
  `tick_id` bigint,
  `killer` bigint,
  `victim` bigint,
  `weapon` smallint,
  `assister` bigint,
  `is_headshot` boolean,
  `is_wallbang` boolean,
  `penetrated_objects` integer
) engine=columnstore;

CREATE TABLE `plants` (
  `id` bigint,
  `start_tick` bigint,
  `end_tick` bigint,
  `planter` bigint,
  `succesful` boolean
) engine=columnstore;

CREATE TABLE `defusals` (
  `id` bigint,
  `plant_id` bigint,
  `start_tick` bigint,
  `end_tick` bigint,
  `defuser` bigint,
  `succesful` boolean
) engine=columnstore;

CREATE TABLE `explosions` (
  `id` bigint,
  `plant_id` bigint,
  `tick_id` bigint
) engine=columnstore;
