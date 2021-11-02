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

CREATE TABLE "per_player_game_lag" (
  "lag" bigint,
  "player" varchar(255),
  "match" varchar(255),
  PRIMARY KEY(lag, player, match)
);
