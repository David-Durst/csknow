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

CREATE TABLE "lookers" (
  "index" bigint PRIMARY KEY,
  "tick_id" bigint,
  "looker_pat_id" bigint,
  "looker_player_id" bigint,
  "looked_at_pat_id" bigint,
  "looked_at_player_id" bigint
);
