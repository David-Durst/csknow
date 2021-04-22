package main

import (
	"fmt"
	"github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/common"
	"github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/events"
	"os"
	"path"
	"sort"
)

const minTicks = 30

type RoundTracker struct {
	id int64
	gameID int64
	startTick int64
	endTick int64
	freezeTimeEnd int64
	roundNumber int
	roundEndReason int
	winner int
}

type GrenadeTracker struct {
	id int64
	thrower int64
	grenadeType common.EquipmentType
	throwTick int64
	activeTick int64
	expiredTick int64
	destroyTick int64
	expired bool
	destroyed bool
}

type PlantTracker struct {
	//will reset to not valid at end of round
	id int64
	startTick int64
	endTick int64
	planter int64
	successful bool
}

type DefusalTracker struct {
	//will reset to not valid at end of round
	id int64
	plantID int64
	startTick int64
	endTick int64
	defuser int64
	successful bool
}

type IDState struct {
	nextGame int64
	nextPlayer int64
	nextRound int64
	nextTick int64
	nextPlayerAtTick int64
	nextSpotted int64
	nextWeaponFire int64
	nextKill int64
	nextPlayerHurt int64
	nextGrenade int64
	nextGrenadeTrajectory int64
	nextPlayerFlashed int64
	nextPlant int64
	nextDefusal int64
	nextExplosion int64
}

type SourceTarget struct {
	source, target int64
}

func getPlayerBySteamID(playersTracker * map[uint64]int64, player * common.Player) int64 {
	if player == nil {
		return -1
	} else {
		return (*playersTracker)[player.SteamID64]
	}
}

func processFile(unprocessedKey string, idState * IDState, firstRun bool, gameTypeToID map[string]int, gameType string) {
	demFilePath := path.Base(unprocessedKey)
	f, err := os.Open(localDemName)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	p := demoinfocs.NewParser(f)
	defer p.Close()

	// generate fact tables (and save if necessary) after parser init
	if firstRun {
		saveEquipmentFile()
	}

	// create games table if it didn't exist, and append header if first run
	gamesFile, err := os.OpenFile(gamesCSVName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		panic(err)
	}
	defer gamesFile.Close()
	if firstRun {
		gamesFile.WriteString("id,demo_file,demo_tick_rate,game_tick_rate,game_type\n")
	}
	curGameID := idState.nextGame

	// setup trackers for logs that cross multiple events
	curRound := RoundTracker{0,0,0,0,0,0,0,0}
	grenadesTracker := make(map[int64]GrenadeTracker)
	playerToLastFireGrenade := make(map[int64]int64)
	curPlant := PlantTracker{0, 0, 0, 0, false}
	curDefusal := DefusalTracker{0, 0, 0, 0, 0, false}
	playersTracker := make(map[uint64]int64)
	lastFlash := make(map[SourceTarget]int64)
	ticksProcessed := 0
	roundsProcessed := 0

	roundsFile, err := os.Create(localRoundsFile)
	if err != nil {
		panic(err)
	}
	defer roundsFile.Close()
	roundsFile.WriteString("id,game_id,start_tick,end_tick,freeze_time_end,round_number,round_end_reason,winner\n")
	p.RegisterEventHandler(func(e events.RoundStart) {
		if ticksProcessed < minTicks {
			return
		}

		curID := idState.nextRound
		idState.nextRound++
		curRound = RoundTracker{curID, curGameID, idState.nextTick, 0,-1, roundsProcessed, 0, 0}
		roundsProcessed++
	})

	const (
		ctSide = 0
		tSide = 1
		spectator = 2
	)

	p.RegisterEventHandler(func(e events.RoundEnd) {
		if ticksProcessed < minTicks {
			return
		}

		curRound.roundEndReason = int(e.Reason)
		if e.Winner == common.TeamCounterTerrorists {
			curRound.winner = ctSide
		} else if e.Winner == common.TeamTerrorists {
			curRound.winner = tSide
		} else {
			curRound.winner = spectator
		}
		curRound.endTick = idState.nextTick
		roundsFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d,%d,%d,%d\n",
			curRound.id, curRound.gameID, curRound.startTick, curRound.endTick, curRound.freezeTimeEnd,
			curRound.roundNumber, curRound.roundEndReason, curRound.winner,
		))
	})

	p.RegisterEventHandler(func(e events.RoundFreezetimeEnd) {
		if ticksProcessed < minTicks {
			return
		}

		curRound.freezeTimeEnd = idState.nextTick
	})

	playersFile, err := os.Create(localPlayersCSVName)
	if err != nil {
		panic(err)
	}
	defer playersFile.Close()
	playersFile.WriteString("id,game_id,name,steam_id\n")
	p.RegisterEventHandler(func(e events.PlayerConnect) {
		playersFile.WriteString(fmt.Sprintf("%d,%d,%s,%d\n",
			idState.nextPlayer, curGameID, e.Player.Name, e.Player.SteamID64))
		playersTracker[e.Player.SteamID64] = idState.nextPlayer
		idState.nextPlayer++
	})

	ticksFile, err := os.Create(localTicksCSVName)
	if err != nil {
		panic(err)
	}
	defer ticksFile.Close()
	ticksFile.WriteString("id,round_id,game_time,warmup,bomb_carrier,bomb_x,bomb_y,bomb_z\n")

	playerAtTickFile, err := os.Create(localPlayerAtTickCSVName)
	if err != nil {
		panic(err)
	}
	defer playerAtTickFile.Close()
	playerAtTickFile.WriteString("id,player_id,tick_id,pos_x,pos_y,pos_z,view_x,view_y,team,health,armor,has_helmet," +
		"is_alive,is_crouching,is_airborne,remaining_flash_time,active_weapon,main_weapon,primary_bullets_clip," +
		"primary_bullets_reserve,secondary_weapon,secondary_bullets_clip,secondary_bullets_reserve,num_he,num_flash,num_smoke," +
		"num_incendiary,num_molotov,num_decoy,num_zeus,has_defuser,has_bomb,money\n")

	p.RegisterEventHandler(func(e events.FrameDone) {
		// on the first tick save the game state
		if ticksProcessed == 0 {
			header := p.Header()
			gamesFile.WriteString(fmt.Sprintf("%d,%s,%f,%f,%d\n",
				curGameID, demFilePath, (&header).FrameRate(), p.TickRate(), gameTypeToID[gameType]))
			idState.nextGame++
		}
		ticksProcessed++
		gs := p.GameState()
		// skip the first couple seconds as tick number  can have issues with these
		// this should be fine as should just be warmup, which is 3 seconds despite fact I told cs to disable
		if ticksProcessed < minTicks {
			return
		}
		gs.IsWarmupPeriod()
		tickID := idState.nextTick
		var carrierID int64
		carrierID = getPlayerBySteamID(&playersTracker, gs.Bomb().Carrier)
		ticksFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d,%d,%.2f,%.2f,%.2f\n",
			tickID,curRound.id,p.CurrentTime().Milliseconds(), gs.IsWarmupPeriod(), carrierID,
			gs.Bomb().Position().X, gs.Bomb().Position().Y, gs.Bomb().Position().Z))

		players := getPlayers(&p)
		sort.Slice(players, func(i int, j int) bool {
			return players[i].Name < players[j].Name
		})
		for _, player := range players {
			playerAtTickID := idState.nextPlayerAtTick
			idState.nextPlayerAtTick++
			primaryWeapon := -1
			primaryBulletsClip := 0
			primaryBulletsReserve := 0
			secondaryWeapon := -1
			secondaryBulletsClip := 0
			secondaryBulletsReserve := 0
			numHE := 0
			numFlash := 0
			numSmoke := 0
			numIncendiary := 0
			numMolotov := 0
			numDecoy := 0
			numZeus := 0
			hasDefuser := false
			hasBomb := false
			for _, weapon := range player.Weapons() {
				if weapon.Class() == common.EqClassPistols {
					secondaryWeapon = int(weapon.Type)
					secondaryBulletsClip = weapon.AmmoInMagazine()
					secondaryBulletsReserve = weapon.AmmoReserve()
				} else if weapon.Class() == common.EqClassSMG || weapon.Class() == common.EqClassHeavy ||
					weapon.Class() == common.EqClassRifle {
					primaryWeapon = int(weapon.Type)
					primaryBulletsClip = weapon.AmmoInMagazine()
					primaryBulletsReserve = weapon.AmmoReserve()
				} else if weapon.Type == common.EqHE {
					numHE = weapon.AmmoReserve() + weapon.AmmoInMagazine()
				} else if weapon.Type == common.EqFlash {
					numFlash = weapon.AmmoReserve() + weapon.AmmoInMagazine()
				} else if weapon.Type == common.EqSmoke {
					numSmoke = weapon.AmmoReserve() + weapon.AmmoInMagazine()
				} else if weapon.Type == common.EqMolotov {
					numMolotov = weapon.AmmoReserve() + weapon.AmmoInMagazine()
				} else if weapon.Type == common.EqIncendiary {
					numIncendiary = weapon.AmmoReserve() + weapon.AmmoInMagazine()
				} else if weapon.Type == common.EqDecoy {
					numDecoy = weapon.AmmoReserve() + weapon.AmmoInMagazine()
				} else if weapon.Type == common.EqZeus {
					numZeus = weapon.AmmoReserve() + weapon.AmmoInMagazine()
				} else if weapon.Type == common.EqDefuseKit {
					hasDefuser = true
				} else if weapon.Type == common.EqBomb {
					hasBomb = true
				}
			}
			activeWeapon := -1
			if player.ActiveWeapon() != nil {
				activeWeapon = int(player.ActiveWeapon().Type)
			}
			side := spectator
			if player.Team == common.TeamCounterTerrorists {
				side = ctSide
			} else if player.Team == common.TeamTerrorists {
				side = tSide
			}
			playerAtTickFile.WriteString(fmt.Sprintf("%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d," +
				"%d,%d,%d,%f,%d,%d,%d," +
				"%d,%d,%d,%d,%d,%d,%d," +
				"%d,%d,%d,%d,%d,%d,%d\n",
				playerAtTickID, getPlayerBySteamID(&playersTracker, player), tickID, player.Position().X, player.Position().Y,
				player.Position().Z, player.ViewDirectionX(), player.ViewDirectionY(), side, player.Health(), player.Armor(),
				boolToInt(player.HasHelmet()),
				boolToInt(player.IsAlive()), boolToInt(player.IsDucking()), boolToInt(player.IsAirborne()), player.FlashDuration,
				activeWeapon, primaryWeapon, primaryBulletsClip,
				primaryBulletsReserve, secondaryWeapon, secondaryBulletsClip, secondaryBulletsReserve, numHE, numFlash, numSmoke,
				numIncendiary, numMolotov, numDecoy, numZeus, boolToInt(hasBomb), boolToInt(hasDefuser), player.Money()))
		}
		idState.nextTick++
	})


	spottedFile, err := os.Create(localSpottedCSVName)
	if err != nil {
		panic(err)
	}
	defer spottedFile.Close()
	spottedFile.WriteString("id,tick_id,spotted_player,spotter_player,is_spotted\n")

	p.RegisterEventHandler(func(e events.PlayerSpottersChanged) {
		if ticksProcessed < minTicks {
			return
		}


		players := getPlayers(&p)
		sort.Slice(players, func(i int, j int) bool {
			return players[i].Name < players[j].Name
		})
		for _, possibleSpotter := range players {
			curID := idState.nextSpotted
			idState.nextSpotted++
			spottedFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d\n",
				curID, idState.nextTick, getPlayerBySteamID(&playersTracker, e.Spotted), getPlayerBySteamID(&playersTracker, possibleSpotter),
				boolToInt(e.Spotted.IsSpottedBy(possibleSpotter))))
		}
	})

	weaponFireFile, err := os.Create(localWeaponFireCSVName)
	if err != nil {
		panic(err)
	}
	defer weaponFireFile.Close()
	weaponFireFile.WriteString("id,tick_id,shooter,weapon\n")

	p.RegisterEventHandler(func(e events.WeaponFire) {
		if ticksProcessed < minTicks {
			return
		}

		curID := idState.nextWeaponFire
		idState.nextWeaponFire++
		weaponFireFile.WriteString(fmt.Sprintf("%d,%d,%d,%d\n",
			curID, idState.nextTick, getPlayerBySteamID(&playersTracker, e.Shooter), int(e.Weapon.Type)))
	})

	hurtFile, err := os.Create(localHurtCSVName)
	if err != nil {
		panic(err)
	}
	defer hurtFile.Close()
	hurtFile.WriteString("id,tick_id,victim,attacker,weapon,armor_damage,armor,health_damage,health,hit_group\n")

	p.RegisterEventHandler(func(e events.PlayerHurt) {
		if ticksProcessed < minTicks {
			return
		}

		curID := idState.nextPlayerHurt
		idState.nextPlayerHurt++
		hurtFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
			curID, idState.nextTick, getPlayerBySteamID(&playersTracker, e.Player), getPlayerBySteamID(&playersTracker, e.Attacker),
			int(e.Weapon.Type), e.ArmorDamage, e.Armor, e.HealthDamage, e.Health, int(e.HitGroup)))
	})

	killsFile, err := os.Create(localKillsCSVName)
	if err != nil {
		panic(err)
	}
	defer killsFile.Close()
	killsFile.WriteString("id,tick_id,killer,victim,weapon,assister,is_headshot,is_wallbang,penetrated_objects\n")

	p.RegisterEventHandler(func(e events.Kill) {
		if ticksProcessed < minTicks {
			return
		}

		curID := idState.nextKill
		idState.nextKill++
		killsFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
			curID, idState.nextTick, getPlayerBySteamID(&playersTracker, e.Killer), getPlayerBySteamID(&playersTracker, e.Victim),
			int(e.Weapon.Type), getPlayerBySteamID(&playersTracker, e.Assister), boolToInt(e.IsHeadshot), boolToInt(e.IsWallBang()),
			e.PenetratedObjects))
	})

	grenadesFile, err := os.Create(localGrenadesCSVName)
	if err != nil {
		panic(err)
	}
	defer grenadesFile.Close()
	grenadesFile.WriteString("id,thrower,grenade_type,throw_tick,active_tick,expired_tick,destroy_tick\n")

	p.RegisterEventHandler(func(e events.GrenadeProjectileThrow) {
		if ticksProcessed < minTicks {
			return
		}

		curID := idState.nextGrenade
		idState.nextGrenade++

		grenadesTracker[e.Projectile.WeaponInstance.UniqueID()] = GrenadeTracker{curID,
			getPlayerBySteamID(&playersTracker, e.Projectile.Thrower),
			e.Projectile.WeaponInstance.Type,
			idState.nextTick,
			0,
			0,
			0,
			false,
			false,
		}

		if e.Projectile.WeaponInstance.Type == common.EqMolotov ||
			e.Projectile.WeaponInstance.Type == common.EqIncendiary {
			playerToLastFireGrenade[getPlayerBySteamID(&playersTracker, e.Projectile.Thrower)] =
				e.Projectile.WeaponInstance.UniqueID()
		}
	})

	saveGrenade := func(id int64) {
		// molotovs and incendiaries are destoryed before effect ends so only save grenade once
		// both have happened
		curGrenade := grenadesTracker[id]
		if curGrenade.destroyed && curGrenade.expired {
			grenadesFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d,%d,%d\n",
				curGrenade.id, curGrenade.thrower, curGrenade.grenadeType,
				curGrenade.throwTick, curGrenade.activeTick, curGrenade.expiredTick, curGrenade.destroyTick))
			delete(grenadesTracker, id)
		}
	}

	p.RegisterEventHandler(func(e events.HeExplode) {
		if ticksProcessed < minTicks {
			return
		}

		curGrenade := grenadesTracker[e.Grenade.UniqueID()]
		curGrenade.activeTick = idState.nextTick
		curGrenade.expiredTick = idState.nextTick
		curGrenade.expired = true
		grenadesTracker[e.Grenade.UniqueID()] = curGrenade
	})

	p.RegisterEventHandler(func(e events.FlashExplode) {
		if ticksProcessed < minTicks {
			return
		}

		curGrenade := grenadesTracker[e.Grenade.UniqueID()]
		curGrenade.activeTick = idState.nextTick
		curGrenade.expiredTick = idState.nextTick
		curGrenade.expired = true
		grenadesTracker[e.Grenade.UniqueID()] = curGrenade
	})

	p.RegisterEventHandler(func(e events.DecoyStart) {
		if ticksProcessed < minTicks {
			return
		}

		curGrenade := grenadesTracker[e.Grenade.UniqueID()]
		curGrenade.activeTick = idState.nextTick
		grenadesTracker[e.Grenade.UniqueID()] = curGrenade
	})

	p.RegisterEventHandler(func(e events.DecoyExpired) {
		if ticksProcessed < minTicks {
			return
		}

		curGrenade := grenadesTracker[e.Grenade.UniqueID()]
		curGrenade.expiredTick = idState.nextTick
		curGrenade.expired = true
		grenadesTracker[e.Grenade.UniqueID()] = curGrenade
	})

	p.RegisterEventHandler(func(e events.SmokeStart) {
		if ticksProcessed < minTicks {
			return
		}

		curGrenade := grenadesTracker[e.Grenade.UniqueID()]
		curGrenade.activeTick = idState.nextTick
		grenadesTracker[e.Grenade.UniqueID()] = curGrenade
	})

	p.RegisterEventHandler(func(e events.SmokeExpired) {
		if ticksProcessed < minTicks {
			return
		}

		curGrenade := grenadesTracker[e.Grenade.UniqueID()]
		curGrenade.expiredTick = idState.nextTick
		curGrenade.expired = true
		grenadesTracker[e.Grenade.UniqueID()] = curGrenade
	})

	p.RegisterEventHandler(func(e events.InfernoStart) {
		if ticksProcessed < minTicks {
			return
		}

		grenadeUniqueID := playerToLastFireGrenade[getPlayerBySteamID(&playersTracker, e.Inferno.Thrower())]
		curGrenade := grenadesTracker[grenadeUniqueID]
		curGrenade.activeTick = idState.nextTick
		grenadesTracker[grenadeUniqueID] = curGrenade
	})

	p.RegisterEventHandler(func(e events.InfernoExpired) {
		if ticksProcessed < minTicks {
			return
		}

		grenadeUniqueID := playerToLastFireGrenade[getPlayerBySteamID(&playersTracker, e.Inferno.Thrower())]
		curGrenade := grenadesTracker[grenadeUniqueID]
		curGrenade.expiredTick = idState.nextTick
		curGrenade.expired = true
		grenadesTracker[grenadeUniqueID] = curGrenade
		saveGrenade(grenadeUniqueID)
	})

	grenadeTrajectoriesFile, err := os.Create(localGrenadeTrajectoriesCSVName)
	if err != nil {
		panic(err)
	}
	defer grenadeTrajectoriesFile.Close()
	grenadeTrajectoriesFile.WriteString("id,grenade_id,id_per_grenade,pos_x,pos_y,pos_z\n")

	p.RegisterEventHandler(func(e events.GrenadeProjectileDestroy) {
		// he grenade destroy happens when smoke from after explosion fades
		// still some smoke left, but totally visible, when smoke grenade expires
		// fire grenades are destroyed as soon as land, then burn for a while
		if ticksProcessed < minTicks {
			return
		}

		curGrenade := grenadesTracker[e.Projectile.WeaponInstance.UniqueID()]
		curGrenade.destroyTick = idState.nextTick
		curGrenade.destroyed = true
		grenadesTracker[e.Projectile.WeaponInstance.UniqueID()] = curGrenade

		for i := range e.Projectile.Trajectory {
			curTrajectoryID := idState.nextGrenadeTrajectory
			idState.nextGrenadeTrajectory++
			grenadeTrajectoriesFile.WriteString(fmt.Sprintf("%d,%d,%d,%.2f,%.2f,%.2f\n",
				curTrajectoryID, curGrenade.id, i,
				e.Projectile.Trajectory[i].X, e.Projectile.Trajectory[i].Y, e.Projectile.Trajectory[i].Z))
		}

		saveGrenade(e.Projectile.WeaponInstance.UniqueID())
	})

	playerFlashedFile, err := os.Create(localPlayerFlashedCSVName)
	if err != nil {
		panic(err)
	}
	defer playerFlashedFile.Close()
	playerFlashedFile.WriteString("id,grenade_id,tick_id,thrower,victim\n")

	p.RegisterEventHandler(func(e events.PlayerFlashed) {
		if ticksProcessed < minTicks {
			return
		}

		source := getPlayerBySteamID(&playersTracker, e.Attacker)
		target := getPlayerBySteamID(&playersTracker, e.Player)
		lastFlashKey := SourceTarget{source, target}

		if oldTick, ok := lastFlash[lastFlashKey]; ok && oldTick == idState.nextTick {
			return
		}
		lastFlash[lastFlashKey]	= idState.nextTick

		curID := idState.nextPlayerFlashed
		idState.nextPlayerFlashed++
		playerFlashedFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d\n",
			curID, e.Projectile.WeaponInstance.UniqueID(), idState.nextTick, getPlayerBySteamID(&playersTracker, e.Attacker),
			getPlayerBySteamID(&playersTracker, e.Player)))
	})


	plantsFile, err := os.Create(localPlantsCSVName)
	if err != nil {
		panic(err)
	}
	defer plantsFile.Close()
	plantsFile.WriteString("id,start_tick,end_tick,planter,successful\n")

	p.RegisterEventHandler(func(e events.BombPlantBegin) {
		if ticksProcessed < minTicks {
			return
		}

		curID := idState.nextDefusal
		idState.nextDefusal++

		curPlant = PlantTracker{curID,
			idState.nextTick,
			0,
			getPlayerBySteamID(&playersTracker, e.Player),
			false,
		}
	})

	p.RegisterEventHandler(func(e events.BombPlantAborted) {
		if ticksProcessed < minTicks {
			return
		}

		curPlant.endTick = idState.nextTick
		curPlant.successful = false
		plantsFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d\n",
			curPlant.id, curPlant.startTick, curPlant.endTick, curPlant.planter, boolToInt(false)))
	})

	p.RegisterEventHandler(func(e events.BombPlanted) {
		if ticksProcessed < minTicks {
			return
		}

		curPlant.endTick = idState.nextTick
		curPlant.successful = true
		plantsFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d\n",
			curPlant.id, curPlant.startTick, curPlant.endTick, curPlant.planter, boolToInt(true)))
	})

	defusalsFile, err := os.Create(localDefusalsSVName)
	if err != nil {
		panic(err)
	}
	defer defusalsFile.Close()
	defusalsFile.WriteString("id,plant_id,start_tick,end_tick,defuser,successful\n")

	p.RegisterEventHandler(func(e events.BombDefuseStart) {
		if ticksProcessed < minTicks {
			return
		}

		curID := idState.nextDefusal
		idState.nextDefusal++

		curDefusal = DefusalTracker{curID,
			curPlant.id,
			idState.nextTick,
			0,
			getPlayerBySteamID(&playersTracker, e.Player),
			false,
		}
	})

	p.RegisterEventHandler(func(e events.BombDefuseAborted) {
		if ticksProcessed < minTicks {
			return
		}

		curDefusal.endTick = idState.nextTick
		curDefusal.successful = false
		defusalsFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d,%d\n",
			curDefusal.id, curDefusal.plantID, curDefusal.startTick, curDefusal.endTick, curDefusal.defuser, boolToInt(false)))
	})

	p.RegisterEventHandler(func(e events.BombDefused) {
		if ticksProcessed < minTicks {
			return
		}

		curDefusal.endTick = idState.nextTick
		curDefusal.successful = true
		defusalsFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d,%d\n",
			curDefusal.id, curDefusal.plantID, curDefusal.startTick, curDefusal.endTick, curDefusal.defuser, boolToInt(true)))
	})

	explosionsFile, err := os.Create(localExplosionsCSVName)
	if err != nil {
		panic(err)
	}
	defer explosionsFile.Close()
	explosionsFile.WriteString("id,plant_id,tick_id\n")

	p.RegisterEventHandler(func(e events.BombExplode) {
		if ticksProcessed < minTicks {
			return
		}

		curID := idState.nextExplosion
		idState.nextExplosion++

		explosionsFile.WriteString(fmt.Sprintf("%d,%d,%d\n", curID, curPlant.id, idState.nextTick))
	})

	err = p.ParseToEnd()
	if err != nil {
		panic(err)
	}
}

func getPlayers(p *demoinfocs.Parser) []*common.Player {
	return (*p).GameState().Participants().Playing()
}

func teamToNum(team common.Team) int {
	switch team {
	case common.TeamUnassigned:
		return 0
	case common.TeamSpectators:
		return 1
	case common.TeamTerrorists:
		return 2
	case common.TeamCounterTerrorists:
		return 3
	}
	return -1
}

func boolToInt(b bool) int {
	if b {
		return 1
	} else {
		return 0
	}
}

