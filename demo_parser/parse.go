package main

import (
	"fmt"
	"github.com/golang/geo/r3"
	"github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/common"
	"github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/events"
	"os"
	"path"
	"sort"
	"strconv"
)

type RoundTracker struct {
	valid bool
	id int64
	gameID int64
	startTick int64
	endTick int64
	warmup bool
	freezeTimeEnd int64
	roundNumber int
	roundEndReason int
	winner int
	tWins int
	ctWins int
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
	trajectory []r3.Vector
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

func getPlayerBySteamID(playersTracker * map[int]int64, player * common.Player) int64 {
	if player == nil {
		return -1
	} else {
		return (*playersTracker)[player.UserID]
	}
}

const (
	ctSide = 0
	tSide = 1
	spectator = 2
)

func finishGarbageRound(round * RoundTracker, idState IDState, tWins int, ctWins int) {
	round.endTick = idState.nextTick - 1
	round.warmup = true
	round.roundEndReason = -1
	round.winner = spectator
	round.tWins = tWins
	round.ctWins = ctWins
}

func processFile(unprocessedKey string, idState * IDState, firstRun bool, gameType int) {
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
	flags := os.O_CREATE|os.O_WRONLY
	if firstRun {
		flags = flags | os.O_TRUNC
	} else {
		flags = flags | os.O_APPEND
	}
	gamesFile, err := os.OpenFile(gamesCSVName, flags, 0644)
	if err != nil {
		panic(err)
	}
	defer gamesFile.Close()
	if firstRun {
		gamesFile.WriteString("id,demo_file,demo_tick_rate,game_tick_rate,game_type\n")
	}
	curGameID := idState.nextGame

	// setup trackers for logs that cross multiple events
	curRound := RoundTracker{false, 0,0,0,0,false, 0,0,0,0, 0, 0}
	// save finished rounds, write them at end so can update warmups if necessary
	var finishedRounds []RoundTracker
	// creating list as flashes thrown back to back will have same id.
	// this could introduce bugs if flashes fuse is impacted by factor other than time thrown, but don't think that is case right now
	grenadesTracker := make(map[int64][]GrenadeTracker)
	lastFlashExplosion := make(map[int64]GrenadeTracker)
	playerToLastFireGrenade := make(map[int64]int64)
	curPlant := PlantTracker{0, 0, 0, 0, false}
	curDefusal := DefusalTracker{0, 0, 0, 0, 0, false}
	playersTracker := make(map[int]int64)
	lastFlash := make(map[SourceTarget]int64)
	ticksProcessed := 0
	roundsProcessed := 0

	roundsFile, err := os.Create(localRoundsCSVName)
	if err != nil {
		panic(err)
	}
	defer roundsFile.Close()
	roundsFile.WriteString("id,game_id,start_tick,end_tick,warmup,freeze_time_end,round_number,round_end_reason,winner,t_wins,ct_wins\n")

	ctWins := 0
	tWins := 0
	p.RegisterEventHandler(func(e events.RoundStart) {
		// can have a round start at end of game, ignore them
		if roundsProcessed > 10 && p.GameState().TeamCounterTerrorists().Score() == 0 &&
			p.GameState().TeamTerrorists().Score() == 0 {
			return
		}
		// warmup can end wihtout a roundend call, so save repeated round starts
		// restarts also seem to start without a round end (like live on 3) - set all to warmup
		if curRound.valid {
			finishGarbageRound(&curRound, *idState, tWins, ctWins)
			finishedRounds = append(finishedRounds, curRound)
		}

		curID := idState.nextRound
		idState.nextRound++
		curRound = RoundTracker{true, curID, curGameID, idState.nextTick, 0, false, -1, roundsProcessed, 0, 0, 0, 0}
		roundsProcessed++
	})

	p.RegisterEventHandler(func(e events.RoundEnd) {
		// skip round ends on first tick, these are worthless
		if idState.nextTick == 0 {
			return
		}
		curRound.roundEndReason = int(e.Reason)
		if e.Winner == common.TeamCounterTerrorists {
			curRound.winner = ctSide
			ctWins++
		} else if e.Winner == common.TeamTerrorists {
			curRound.winner = tSide
			tWins++
		} else {
			curRound.winner = spectator
		}
		if tWins + ctWins == 15 {
			fmt.Printf("change sides\n")
			oldCTWins := ctWins
			ctWins = tWins
			tWins = oldCTWins
		}
		curRound.endTick = idState.nextTick
		// handle demos that start after first round start or just miss a round start event
		if !curRound.valid {
			curRound.id = idState.nextRound
			idState.nextRound++
			curRound.roundNumber = roundsProcessed
			roundsProcessed++
			// can have short pre-warmup round that ends without round start, so mark that as warmup
			if curRound.roundNumber == 0 {
				curRound.warmup = true
			} else {
				// if normal state but just missed round start event, we'll just make this round really short
				curRound.startTick = curRound.endTick
			}
		}
		curRound.tWins = tWins
		curRound.ctWins = ctWins
		finishedRounds = append(finishedRounds, curRound)
		curRound.valid = false
		curRound.warmup = false
	})

	p.RegisterEventHandler(func(e events.RoundFreezetimeEnd) {
		curRound.freezeTimeEnd = idState.nextTick
	})

	playersFile, err := os.Create(localPlayersCSVName)
	if err != nil {
		panic(err)
	}
	defer playersFile.Close()
	playersFile.WriteString("id,game_id,name,steam_id\n")
	// add -1 player at start of first players file
	if idState.nextPlayer == 0 {
		playersFile.WriteString("-1,\\N,invalid,0\n")
	}

	ticksFile, err := os.Create(localTicksCSVName)
	if err != nil {
		panic(err)
	}
	defer ticksFile.Close()
	ticksFile.WriteString("id,round_id,game_time,demo_tick_number,game_tick_number,bomb_carrier,bomb_x,bomb_y,bomb_z\n")

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
				curGameID, demFilePath, (&header).FrameRate(), p.TickRate(), gameType))
			idState.nextGame++
		}
		ticksProcessed++
		gs := p.GameState()
		if gs.IsWarmupPeriod() {
			curRound.warmup = true
		}
		tickID := idState.nextTick
		players := getPlayers(&p)
		sort.Slice(players, func(i int, j int) bool {
			return players[i].Name < players[j].Name
		})
		for _, player := range players {
			if _, ok := playersTracker[player.UserID]; !ok {
				playersFile.WriteString(fmt.Sprintf("%d,%d,%s,%d\n",
					idState.nextPlayer, curGameID, player.Name, player.SteamID64))
				playersTracker[player.UserID] = idState.nextPlayer
				idState.nextPlayer++
			}
		}
		var carrierID int64
		carrierID = getPlayerBySteamID(&playersTracker, gs.Bomb().Carrier)
		ticksFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d,%d,%.2f,%.2f,%.2f\n",
			tickID,curRound.id,p.CurrentTime().Milliseconds(), p.CurrentFrame(), gs.IngameTick(),
			carrierID, gs.Bomb().Position().X, gs.Bomb().Position().Y, gs.Bomb().Position().Z))


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
		curID := idState.nextPlayerHurt
		idState.nextPlayerHurt++
		hitGroup := int(e.HitGroup)
		if hitGroup < 0 || (hitGroup > 7 && hitGroup != 10) {
			hitGroup = -1
		}
		hurtFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
			curID, idState.nextTick, getPlayerBySteamID(&playersTracker, e.Player), getPlayerBySteamID(&playersTracker, e.Attacker),
			int(e.Weapon.Type), e.ArmorDamage, e.Armor, e.HealthDamage, e.Health, hitGroup))
	})

	killsFile, err := os.Create(localKillsCSVName)
	if err != nil {
		panic(err)
	}
	defer killsFile.Close()
	killsFile.WriteString("id,tick_id,killer,victim,weapon,assister,is_headshot,is_wallbang,penetrated_objects\n")

	p.RegisterEventHandler(func(e events.Kill) {
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

	grenadeTrajectoriesFile, err := os.Create(localGrenadeTrajectoriesCSVName)
	if err != nil {
		panic(err)
	}
	defer grenadeTrajectoriesFile.Close()
	grenadeTrajectoriesFile.WriteString("id,grenade_id,id_per_grenade,pos_x,pos_y,pos_z\n")

	p.RegisterEventHandler(func(e events.GrenadeProjectileThrow) {
		curID := idState.nextGrenade
		idState.nextGrenade++

		grenadesTracker[e.Projectile.WeaponInstance.UniqueID()] =
			append(grenadesTracker[e.Projectile.WeaponInstance.UniqueID()], GrenadeTracker{curID,
				getPlayerBySteamID(&playersTracker, e.Projectile.Thrower),
				e.Projectile.WeaponInstance.Type,
				idState.nextTick,
				0,
				0,
				0,
				false,
				false,
				nil,
			})

		if e.Projectile.WeaponInstance.Type == common.EqMolotov ||
			e.Projectile.WeaponInstance.Type == common.EqIncendiary {
			playerToLastFireGrenade[getPlayerBySteamID(&playersTracker, e.Projectile.Thrower)] =
				e.Projectile.WeaponInstance.UniqueID()
		}
	})

	saveGrenade := func(id int64) {
		// molotovs and incendiaries are destoryed before effect ends so only save grenade once
		// both have happened
		curGrenade := grenadesTracker[id][0]
		if curGrenade.destroyed && curGrenade.expired {
			activeString := strconv.FormatInt(curGrenade.activeTick, 10)
			if curGrenade.activeTick == 0 {
				activeString = "\\N"
			}
			expiredString := strconv.FormatInt(curGrenade.expiredTick, 10)
			if curGrenade.expiredTick == 0 {
				expiredString = "\\N"
			}
			destoryString := strconv.FormatInt(curGrenade.destroyTick, 10)
			if curGrenade.destroyTick == 0 {
				destoryString = "\\N"
			}
			grenadesFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%s,%s,%s\n",
				curGrenade.id, curGrenade.thrower, curGrenade.grenadeType,
				curGrenade.throwTick, activeString, expiredString, destoryString))

			for i := range curGrenade.trajectory {
				curTrajectoryID := idState.nextGrenadeTrajectory
				idState.nextGrenadeTrajectory++
				grenadeTrajectoriesFile.WriteString(fmt.Sprintf("%d,%d,%d,%.2f,%.2f,%.2f\n",
					curTrajectoryID, curGrenade.id, i,
					curGrenade.trajectory[i].X, curGrenade.trajectory[i].Y, curGrenade.trajectory[i].Z))
			}

			grenadesTracker[id] = grenadesTracker[id][1:]
			if len(grenadesTracker[id]) == 0 {
				delete(grenadesTracker, id)
			}
		}
	}

	p.RegisterEventHandler(func(e events.HeExplode) {
		if _, ok := grenadesTracker[e.Grenade.UniqueID()]; !ok {
			return
		}
		curGrenade := grenadesTracker[e.Grenade.UniqueID()][0]
		curGrenade.activeTick = idState.nextTick
		curGrenade.expiredTick = idState.nextTick
		curGrenade.expired = true
		grenadesTracker[e.Grenade.UniqueID()][0] = curGrenade
	})

	p.RegisterEventHandler(func(e events.FlashExplode) {
		if _, ok := grenadesTracker[e.Grenade.UniqueID()]; !ok {
			return
		}
		curGrenade := grenadesTracker[e.Grenade.UniqueID()][0]
		curGrenade.activeTick = idState.nextTick
		curGrenade.expiredTick = idState.nextTick
		curGrenade.expired = true
		grenadesTracker[e.Grenade.UniqueID()][0] = curGrenade
		lastFlashExplosion[e.Grenade.UniqueID()] = curGrenade
	})

	p.RegisterEventHandler(func(e events.DecoyStart) {
		if _, ok := grenadesTracker[e.Grenade.UniqueID()]; !ok {
			return
		}
		curGrenade := grenadesTracker[e.Grenade.UniqueID()][0]
		curGrenade.activeTick = idState.nextTick
		grenadesTracker[e.Grenade.UniqueID()][0] = curGrenade
	})

	p.RegisterEventHandler(func(e events.DecoyExpired) {
		if _, ok := grenadesTracker[e.Grenade.UniqueID()]; !ok {
			return
		}
		curGrenade := grenadesTracker[e.Grenade.UniqueID()][0]
		curGrenade.expiredTick = idState.nextTick
		curGrenade.expired = true
		grenadesTracker[e.Grenade.UniqueID()][0] = curGrenade
	})

	p.RegisterEventHandler(func(e events.SmokeStart) {
		if _, ok := grenadesTracker[e.Grenade.UniqueID()]; !ok {
			return
		}
		curGrenade := grenadesTracker[e.Grenade.UniqueID()][0]
		curGrenade.activeTick = idState.nextTick
		grenadesTracker[e.Grenade.UniqueID()][0] = curGrenade
	})

	p.RegisterEventHandler(func(e events.SmokeExpired) {
		if _, ok := grenadesTracker[e.Grenade.UniqueID()]; !ok {
			return
		}
		curGrenade := grenadesTracker[e.Grenade.UniqueID()][0]
		curGrenade.expiredTick = idState.nextTick
		curGrenade.expired = true
		grenadesTracker[e.Grenade.UniqueID()][0] = curGrenade
	})

	p.RegisterEventHandler(func(e events.InfernoStart) {
		grenadeUniqueID := playerToLastFireGrenade[getPlayerBySteamID(&playersTracker, e.Inferno.Thrower())]
		if _, ok := grenadesTracker[grenadeUniqueID]; !ok {
			return
		}
		curGrenade := grenadesTracker[grenadeUniqueID][0]
		curGrenade.activeTick = idState.nextTick
		grenadesTracker[grenadeUniqueID][0] = curGrenade
	})

	p.RegisterEventHandler(func(e events.InfernoExpired) {
		grenadeUniqueID := playerToLastFireGrenade[getPlayerBySteamID(&playersTracker, e.Inferno.Thrower())]
		if _, ok := grenadesTracker[grenadeUniqueID]; !ok {
			return
		}
		curGrenade := grenadesTracker[grenadeUniqueID][0]
		curGrenade.expiredTick = idState.nextTick
		curGrenade.expired = true
		grenadesTracker[grenadeUniqueID][0] = curGrenade
		saveGrenade(grenadeUniqueID)
	})

	p.RegisterEventHandler(func(e events.GrenadeProjectileDestroy) {
		if _, ok := grenadesTracker[e.Projectile.WeaponInstance.UniqueID()]; !ok {
			return
		}
		// he grenade destroy happens when smoke from after explosion fades
		// still some smoke left, but totally visible, when smoke grenade expires
		// fire grenades are destroyed as soon as land, then burn for a while
		//fmt.Printf("destroying uid: %d\n", e.Projectile.WeaponInstance.UniqueID())
		curGrenade := grenadesTracker[e.Projectile.WeaponInstance.UniqueID()][0]
		curGrenade.destroyTick = idState.nextTick
		curGrenade.destroyed = true
		curGrenade.trajectory = e.Projectile.Trajectory
		grenadesTracker[e.Projectile.WeaponInstance.UniqueID()][0] = curGrenade
		saveGrenade(e.Projectile.WeaponInstance.UniqueID())
	})

	playerFlashedFile, err := os.Create(localPlayerFlashedCSVName)
	if err != nil {
		panic(err)
	}
	defer playerFlashedFile.Close()
	playerFlashedFile.WriteString("id,tick_id,grenade_id,thrower,victim\n")

	p.RegisterEventHandler(func(e events.PlayerFlashed) {
		source := getPlayerBySteamID(&playersTracker, e.Attacker)
		target := getPlayerBySteamID(&playersTracker, e.Player)
		// this handles player flashed event firing twice
		lastFlashKey := SourceTarget{source, target}

		if oldTick, ok := lastFlash[lastFlashKey]; ok && oldTick == idState.nextTick {
			return
		}
		lastFlash[lastFlashKey]	= idState.nextTick

		curID := idState.nextPlayerFlashed
		idState.nextPlayerFlashed++
		// filter for flashes whose explosion wasn't recorded
		if grenade, ok := lastFlashExplosion[e.Projectile.WeaponInstance.UniqueID()]; ok {
			playerFlashedFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d\n",
				curID, idState.nextTick, grenade.id, getPlayerBySteamID(&playersTracker, e.Attacker),
				getPlayerBySteamID(&playersTracker, e.Player)))
		}
	})


	plantsFile, err := os.Create(localPlantsCSVName)
	if err != nil {
		panic(err)
	}
	defer plantsFile.Close()
	plantsFile.WriteString("id,start_tick,end_tick,planter,successful\n")

	p.RegisterEventHandler(func(e events.BombPlantBegin) {
		curID := idState.nextPlant
		idState.nextPlant++

		curPlant = PlantTracker{curID,
			idState.nextTick,
			0,
			getPlayerBySteamID(&playersTracker, e.Player),
			false,
		}
	})

	p.RegisterEventHandler(func(e events.BombPlantAborted) {
		curPlant.endTick = idState.nextTick
		curPlant.successful = false
		plantsFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d\n",
			curPlant.id, curPlant.startTick, curPlant.endTick, curPlant.planter, boolToInt(false)))
	})

	p.RegisterEventHandler(func(e events.BombPlanted) {
		curPlant.endTick = idState.nextTick
		curPlant.successful = true
		plantsFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d\n",
			curPlant.id, curPlant.startTick, curPlant.endTick, curPlant.planter, boolToInt(true)))
	})

	defusalsFile, err := os.Create(localDefusalsCSVName)
	if err != nil {
		panic(err)
	}
	defer defusalsFile.Close()
	defusalsFile.WriteString("id,plant_id,start_tick,end_tick,defuser,successful\n")

	p.RegisterEventHandler(func(e events.BombDefuseStart) {
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
		curDefusal.endTick = idState.nextTick
		curDefusal.successful = false
		defusalsFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d,%d\n",
			curDefusal.id, curDefusal.plantID, curDefusal.startTick, curDefusal.endTick, curDefusal.defuser, boolToInt(false)))
	})

	p.RegisterEventHandler(func(e events.BombDefused) {
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
		curID := idState.nextExplosion
		idState.nextExplosion++

		explosionsFile.WriteString(fmt.Sprintf("%d,%d,%d\n", curID, curPlant.id, idState.nextTick))
	})

	err = p.ParseToEnd()
	if err != nil {
		fmt.Printf("Error in parsing. T score %d, CT score %d, progress: %f, error:\n %s\n",
			p.GameState().TeamTerrorists().Score(), p.GameState().TeamCounterTerrorists().Score(), p.Progress(), err.Error())
	}
	// update warmups
	lastWarmupRound := -1
	for i := range finishedRounds {
		if finishedRounds[i].warmup {
			lastWarmupRound = i
		}
	}
	for i := 0; i < lastWarmupRound; i++ {
		finishedRounds[i].warmup = true
	}
	// if extra round at end, make it a warmup round and finish it
	if curRound.valid {
		finishGarbageRound(&curRound, *idState, tWins, ctWins)
		finishedRounds = append(finishedRounds, curRound)
	}
	for _, round := range finishedRounds {
		roundsFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
			round.id, round.gameID, round.startTick, round.endTick, boolToInt(round.warmup), round.freezeTimeEnd,
			round.roundNumber, round.roundEndReason, round.winner, round.tWins, round.ctWins
		))
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

