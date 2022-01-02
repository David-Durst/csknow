CREATE TABLE "hand_visibility" (
  "index" bigint PRIMARY KEY,
  "spotter" varchar(255),
  "spotted" varchar(255),
  "start_game_tick" bigint,
  "end_game_tick" bigint,
  "spotter_id" bigint,
  "spotted_id" bigint,
  "demo" varchar(255),
  "hacking" boolean
);

CREATE TABLE "visibilities" (
  "spotter" varchar(255),
  "spotted" varchar(255),
  "start_game_tick" bigint,
  "end_game_tick" bigint,
  "spotter_id" bigint,
  "spotted_id" bigint,
  "demo" varchar(255),
  "hacking" int,
  "start_frame_num" bigint,
  "end_frame_num" bigint,
  "color" varchar(255),
  PRIMARY KEY(spotter, spotted, start_game_tick, demo)
);

CREATE TABLE "visibilities_unadjusted" (
  "spotter" varchar(255),
  "spotted" varchar(255),
  "start_game_tick" bigint,
  "end_game_tick" bigint,
  "spotter_id" bigint,
  "spotted_id" bigint,
  "demo" varchar(255),
  "hacking" int,
  "start_frame_num" bigint,
  "end_frame_num" bigint,
  "color" varchar(255),
  PRIMARY KEY(spotter, spotted, start_game_tick, demo)
);

CREATE TABLE "lookers" (
  "index" bigint PRIMARY KEY,
  "tick_id" bigint,
  "looker_pat_id" bigint,
  "looker_player_id" bigint,
  "looked_at_pat_id" bigint,
  "looked_at_player_id" bigint
);

CREATE TABLE "actions" (
  "action_name" varchar(255),
  "action_number" bigint,
  "game_tick_number" bigint,
  "demo" varchar(255),
  "player" varchar(255),
  PRIMARY KEY(action_number, game_tick_number, demo, player)
);

CREATE TABLE "cover_origins" (
  "index" bigint PRIMARY KEY,
  "x" double precision,
  "y" double precision,
  "z" double precision
);

CREATE TABLE "cover_edges" (
  "index" bigint PRIMARY KEY,
  "origin_id" bigint,
  "cluster_id" bigint,
  "minX" double precision,
  "minY" double precision,
  "minZ" double precision,
  "maxX" double precision,
  "maxY" double precision,
  "maxZ" double precision
);

CREATE TABLE "nearest_origin" (
  "index" bigint PRIMARY KEY,
  "tick_id" bigint,
  "pat_id" bigint,
  "player_id" bigint,
  "origin_id" bigint
);

CREATE TABLE "player_in_cover_edge" (
  "index" bigint PRIMARY KEY,
  "tick_id" bigint,
  "looker_pat_id" bigint,
  "looker_player_id" bigint,
  "looked_at_pat_id" bigint,
  "looked_at_player_id" bigint,
  "nearest_origin_id" bigint,
  "cover_edge_id" bigint
);

CREATE TABLE "player_looking_at_cover_edge" (
  "index" bigint PRIMARY KEY,
  "tick_id" bigint,
  "cur_pat_id" bigint,
  "cur_player_id" bigint,
  "looker_pat_id" bigint,
  "looker_player_id" bigint,
  "nearest_origin_id" bigint,
  "cover_edge_id" bigint
);
