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
	valid bool
	id int64
	startTick int64
	endTick int64
}

type DefusalTracker struct {
	//will reset to not valid at end of round
	valid bool
	id int64
	startTick int64
	// not tracking destroyTick since save on endTick
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
	nextFlashed int64
	nextPlant int64
	nextDefusal int64
	nextExplosion int64
}


func processFile(unprocessedKey string, idState * IDState, firstRun bool) {
	demFilePath := path.Base(unprocessedKey)
	f, err := os.Open(localDemName)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	positionFile, err := os.Create(localPositionCSVName)
	if err != nil {
		panic(err)
	}
	defer positionFile.Close()
	positionFile.WriteString("demo tick number,ingame tick,match started,game phase,rounds played,is warmup,round start,round end,round end reason,freeze time ended,t score,ct score,num players")
	for i := 0; i < 10; i++ {
		positionFile.WriteString(fmt.Sprintf(
			",player %d name,player %d team,player %d x position,player %d y position,player %d z position" +
				",player %d x view direction,player %d y view direction,player %d is alive,player %d is blinded",
				i, i, i, i, i, i, i, i, i))
	}
	positionFile.WriteString(fmt.Sprintf(",demo file\n"))

	p := demoinfocs.NewParser(f)
	defer p.Close()

	// generate fact tables (and save if necessary) after parser init
	equipmentToName := makeEquipmentToName()
	if firstRun {
		saveEquipmentFile(equipmentToName)
	}

	// create games table if it didn't exist, and append header if first run
	gamesFile, err := os.OpenFile(gamesCSVName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		panic(err)
	}
	defer gamesFile.Close()
	if firstRun {
		gamesFile.WriteString("id,demo_file,demo_tick_rate,game_tick_rate\n")
	}
	curGameID := idState.nextGame

	// flags from other event handlers to frame done handler
	roundStart := 0
	roundEndReason := -1
	roundEnd := 0
	freezeTime := 0

	// setup trackers for logs that cross multiple events
	grenadesTracker := make(map[int64]GrenadeTracker)
	playerToLastFireGrenade := make(map[int64]int64)
	//curPlant := PlantTracker{false, 0, 0, 0}
	//curDefusal := DefusalTracker{false, 0, 0}
	playersTracker := make(map[uint64]int64)

	p.RegisterEventHandler(func(e events.RoundStart) {
		roundStart = 1
		freezeTime = 1
	})

	p.RegisterEventHandler(func(e events.RoundEnd) {
		roundEnd = 1
		switch e.Reason {
		case events.RoundEndReasonTargetBombed:
			roundEndReason = 1
		case events.RoundEndReasonVIPEscaped:
			roundEndReason = 2
		case events.RoundEndReasonVIPKilled:
			roundEndReason = 3
		case events.RoundEndReasonTerroristsEscaped:
			roundEndReason = 4
		case events.RoundEndReasonCTStoppedEscape:
			roundEndReason = 5
		case events.RoundEndReasonTerroristsStopped:
			roundEndReason = 6
		case events.RoundEndReasonBombDefused:
			roundEndReason = 7
		case events.RoundEndReasonCTWin:
			roundEndReason = 8
		case events.RoundEndReasonTerroristsWin:
			roundEndReason = 9
		case events.RoundEndReasonDraw:
			roundEndReason = 10
		case events.RoundEndReasonHostagesRescued:
			roundEndReason = 11
		case events.RoundEndReasonTargetSaved:
			roundEndReason = 12
		case events.RoundEndReasonHostagesNotRescued:
			roundEndReason = 13
		case events.RoundEndReasonTerroristsNotEscaped:
			roundEndReason = 14
		case events.RoundEndReasonVIPNotEscaped:
			roundEndReason = 15
		case events.RoundEndReasonGameStart:
			roundEndReason = 16
		case events.RoundEndReasonTerroristsSurrender:
			roundEndReason = 17
		case events.RoundEndReasonCTSurrender:
			roundEndReason = 18
		}
	})

	p.RegisterEventHandler(func(e events.RoundFreezetimeEnd) {
		freezeTime = 0
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

	ticksProcessed := 0
	p.RegisterEventHandler(func(e events.FrameDone) {
		// on the first tick save the game state
		if ticksProcessed == 0 {
			header := p.Header()
			gamesFile.WriteString(fmt.Sprintf("%d,%s,%f,%f\n", curGameID, demFilePath, (&header).FrameRate(), p.TickRate()))
			idState.nextGame++
		}
		ticksProcessed++
		gs := p.GameState()
		players := getPlayers(&p)
		// skip the first couple seconds as tick number  can have issues with these
		// this should be fine as should just be warmup, which is 3 seconds despite fact I told cs to disable
		if ticksProcessed < minTicks {
			return
		}
		matchStarted := 0
		if gs.IsMatchStarted() {
			matchStarted = 1
		}
		isWarmup := 0
		if gs.IsWarmupPeriod() {
			isWarmup = 1
		}
		positionFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,", p.CurrentFrame(), gs.IngameTick(), matchStarted, gs.GamePhase(), gs.TotalRoundsPlayed(), isWarmup,
			roundStart, roundEnd, roundEndReason, freezeTime, gs.TeamTerrorists().Score(), gs.TeamCounterTerrorists().Score(), len(players)))
		sort.Slice(players, func(i int, j int) bool {
			return players[i].Name < players[j].Name
		})
		for i := 0; i < 10; i++ {
			if i >= len(players) {
				positionFile.WriteString(",,,,,")
			} else {
				isAlive := 0
				if players[i].IsAlive() {
					isAlive = 1
				}
				isBlinded := 0
				if players[i].IsBlinded() {
					isBlinded = 1
				}
				positionFile.WriteString(fmt.Sprintf("%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d",
					players[i].Name, teamToNum(players[i].Team), players[i].Position().X, players[i].Position().Y,
					players[i].Position().Z, players[i].ViewDirectionX(), players[i].ViewDirectionY(),
					isAlive, isBlinded))
			}
			positionFile.WriteString(",")
		}
		positionFile.WriteString(demFilePath + "\n")
		roundStart = 0
		roundEnd = 0
		roundEndReason = -1
		idState.nextTick++
	})


	spottedFile, err := os.Create(localSpottedCSVName)
	if err != nil {
		panic(err)
	}
	defer spottedFile.Close()
	spottedFile.WriteString("spotted player")
	for i := 0; i < 10; i++ {
		spottedFile.WriteString(fmt.Sprintf(
			",player %d name,player %d spotter", i, i))
	}
	spottedFile.WriteString(fmt.Sprintf("tick number,demo file\n"))

	p.RegisterEventHandler(func(e events.PlayerSpottersChanged) {
		if ticksProcessed < minTicks {
			return
		}

		players := getPlayers(&p)

		sort.Slice(players, func(i int, j int) bool {
			return players[i].Name < players[j].Name
		})
		spottedFile.WriteString(fmt.Sprintf("%s,", e.Spotted.Name))
		for i := 0; i < 10; i++ {
			if i >= len(players) {
				spottedFile.WriteString(",0,")
			} else {
				spottedFlag := 0
				if e.Spotted.IsSpottedBy(players[i]) {
					spottedFlag = 1
				}
				spottedFile.WriteString(fmt.Sprintf("%s,%d,", players[i].Name, spottedFlag))
			}
		}
		spottedFile.WriteString(fmt.Sprintf("%d,%s\n", p.CurrentFrame(), demFilePath))
	})

	weaponFireFile, err := os.Create(localWeaponFireCSVName)
	if err != nil {
		panic(err)
	}
	defer weaponFireFile.Close()
	weaponFireFile.WriteString("shooter,weapon,tick number,demo file\n")

	p.RegisterEventHandler(func(e events.WeaponFire) {
		if ticksProcessed < minTicks {
			return
		}

		weaponFireFile.WriteString(fmt.Sprintf("%s,%s,%d,%s\n",
			e.Shooter.Name, e.Weapon.String(), p.CurrentFrame(), demFilePath))
	})

	hurtFile, err := os.Create(localHurtCSVName)
	if err != nil {
		panic(err)
	}
	defer hurtFile.Close()
	hurtFile.WriteString("victim name,armor damage,armor,health damage,health,attacker,weapon,tick number,demo file\n")

	p.RegisterEventHandler(func(e events.PlayerHurt) {
		if ticksProcessed < minTicks {
			return
		}

		attackerName := "n/a"
		if e.Attacker != nil {
			attackerName =  e.Attacker.Name
		}
		weaponName := "n/a"
		if e.Weapon != nil {
			weaponName = e.Weapon.String()
		}
		hurtFile.WriteString(fmt.Sprintf("%s,%d,%d,%d,%d,%s,%s,%d,%s\n",
			e.Player.Name, e.ArmorDamage, e.Armor, e.HealthDamage, e.Health, attackerName, weaponName,
			p.CurrentFrame(), demFilePath))
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
			playersTracker[e.Projectile.Thrower.SteamID64],
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
			playerToLastFireGrenade[playersTracker[e.Projectile.Thrower.SteamID64]] =
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

		fmt.Printf("id %d\n", e.Grenade.UniqueID())
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

		grenadeUniqueID := playerToLastFireGrenade[playersTracker[e.Inferno.Thrower().SteamID64]]
		curGrenade := grenadesTracker[grenadeUniqueID]
		curGrenade.activeTick = idState.nextTick
		grenadesTracker[grenadeUniqueID] = curGrenade
	})

	/*
	p.RegisterEventHandler(func(e events.FireGrenadeExpired) {
		if ticksProcessed < minTicks {
			return
		}

		fmt.Printf("fire grenade expired on %d\n", idState.nextTick)
		fmt.Printf("fire grenade expired position %s\n", e.Position.String())
		curID := fireTracker[e.Position]
		fmt.Printf("curID %d\n", curID)
		/*
		fmt.Printf("id %d\n", e.Grenade.UniqueID())
		curGrenade := grenadesTracker[e.Grenade.UniqueID()]
		curGrenade.expiredTick = idState.nextTick
		curGrenade.expired = true
		grenadesTracker[e.Grenade.UniqueID()] = curGrenade
		saveGrenade(e.Grenade.UniqueID())
	})
*/

	p.RegisterEventHandler(func(e events.InfernoExpired) {
		if ticksProcessed < minTicks {
			return
		}

		fmt.Printf("inferno grenade expired on %d\n", idState.nextTick)
		fmt.Printf("inferno thrower %d\n", e.Inferno.Thrower().SteamID64)
		grenadeUniqueID := playerToLastFireGrenade[playersTracker[e.Inferno.Thrower().SteamID64]]
		curGrenade := grenadesTracker[grenadeUniqueID]
		curGrenade.expiredTick = idState.nextTick
		curGrenade.expired = true
		grenadesTracker[grenadeUniqueID] = curGrenade
		saveGrenade(grenadeUniqueID)
		/*
			fmt.Printf("id %d\n", e.Grenade.UniqueID())
			curGrenade := grenadesTracker[e.Grenade.UniqueID()]
			curGrenade.expiredTick = idState.nextTick
			curGrenade.expired = true
			grenadesTracker[e.Grenade.UniqueID()] = curGrenade
			saveGrenade(e.Grenade.UniqueID())
		*/
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

		fmt.Printf("weapon instance unqiue id %d\n", e.Projectile.WeaponInstance.UniqueID())
		fmt.Printf("grenade id %d\n", e.Projectile.Entity.ID())
		fmt.Printf("porjectile unique id %d\n", e.Projectile.UniqueID())
		curGrenade := grenadesTracker[e.Projectile.WeaponInstance.UniqueID()]
		curGrenade.destroyTick = idState.nextTick
		curGrenade.destroyed = true
		grenadesTracker[e.Projectile.WeaponInstance.UniqueID()] = curGrenade

		for i := range e.Projectile.Trajectory {
			curTrajectoryID := idState.nextGrenadeTrajectory
			idState.nextGrenadeTrajectory++
			grenadeTrajectoriesFile.WriteString(fmt.Sprintf("%d,%d,%d,%f,%f,%f\n",
				curTrajectoryID, curGrenade.id, i,
				e.Projectile.Trajectory[i].X, e.Projectile.Trajectory[i].Y, e.Projectile.Trajectory[i].Z))
		}

		saveGrenade(e.Projectile.WeaponInstance.UniqueID())
	})


	killsFile, err := os.Create(localKillsCSVName)
	if err != nil {
		panic(err)
	}
	defer killsFile.Close()
	killsFile.WriteString("killer,victim,weapon,assister,is headshot,is wallbang,penetrated objects,tick number,demo file\n")

	p.RegisterEventHandler(func(e events.Kill) {
		if ticksProcessed < minTicks {
			return
		}

		killerName := "n/a"
		if e.Killer != nil {
			killerName =  e.Killer.Name
		}
		assisterName := "n/a"
		if e.Assister != nil {
			assisterName =  e.Assister.Name
		}
		weaponName := "n/a"
		if e.Weapon != nil {
			weaponName = e.Weapon.String()
		}
		killsFile.WriteString(fmt.Sprintf("%s,%s,%s,%s,%d,%d,%d,%d,%s\n",
			killerName, e.Victim.Name, weaponName, assisterName,
			boolToInt(e.IsHeadshot), boolToInt(e.IsWallBang()), e.PenetratedObjects,
			p.CurrentFrame(), demFilePath))
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

