package internal

import (
	"fmt"
	c "github.com/David-Durst/csknow/demo_parser/internal/constants"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs/common"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs/events"
	"os"
	"path"
	"path/filepath"
	"sort"
)

func addNewRound(idState *IDState, nextTickId RowIndex) {
	// finish last round if one exists
	if roundsTable.len() > 0 {
		roundsTable.tail().finished = true
	}

	roundsTable.append(defaultRound)
	roundsTable.tail().id = idState.nextRound
	roundsTable.tail().gameId = curGameRow.id
	roundsTable.tail().startTick = nextTickId
	roundsTable.tail().warmup = true
	roundsTable.tail().roundNumber = roundsTable.len() - 1
	idState.nextRound++
}

func ProcessStructure(unprocessedKey string, localDemName string, idState *IDState, firstRun bool, gameType c.GameType) {
	curGameRow.id = idState.nextGame
	// increment this locally as won't actually record ticks until after past processing sturcture
	nextTickId := idState.nextTick
	fmt.Printf("localDemName: %s\n", localDemName)
	f, err := os.Open(localDemName)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	p := demoinfocs.NewParser(f)
	defer p.Close()

	p.RegisterEventHandler(func(e events.RoundStart) {
		// for now, adding all rounds, so can examine again garbage at end of match
		// initializing as warmup, will fix to non-warmup when get a round end call
		addNewRound(idState, nextTickId)
	})

	p.RegisterEventHandler(func(e events.RoundEnd) {
		// skip round ends on first tick, these are worthless
		if nextTickId == 0 {
			return
		}

		if roundsTable.len() > 0 && !roundsTable.tail().finished {
			roundsTable.tail().roundEndReason = int(e.Reason)
			roundsTable.tail().winner = e.Winner
			roundsTable.tail().endTick = nextTickId
			roundsTable.tail().warmup = false
		} else {
			// handle demos that start after first round start or just miss a round start event
			// assume bogus rounds are warmups
			addNewRound(idState, nextTickId)
		}

		roundsTable.tail().tWins = p.GameState().Team(common.TeamTerrorists).Score()
		roundsTable.tail().ctWins = p.GameState().Team(common.TeamCounterTerrorists).Score()
		roundsTable.tail().finished = true
	})

	p.RegisterEventHandler(func(e events.RoundFreezetimeEnd) {
		// only update once per round round, as can fire duplicate event at end of last round of match
		if roundsTable.tail().freezeTimeEnd == InvalidId {
			roundsTable.tail().freezeTimeEnd = nextTickId
		}
	})

	var playersTracker playersTrackerT
	playersTracker.init()
	if idState.nextPlayer == 0 {
		playersTable.append(playerRow{InvalidId, InvalidId, "invalid", 0})
	}
	p.RegisterEventHandler(func(e events.FrameDone) {
		// on the first tick save the game state
		if nextTickId == idState.nextTick {
			header := p.Header()
			curGameRow.demoFile = path.Base(unprocessedKey)
			curGameRow.demoTickRate = header.FrameRate()
			curGameRow.gameTickRate = p.TickRate()
			curGameRow.mapName = header.MapName
			curGameRow.gameType = gameType
			idState.nextGame++
		}

		// add all new players
		players := getPlayers(&p)
		sort.Slice(players, func(i int, j int) bool {
			return players[i].Name < players[j].Name
		})
		for _, player := range players {
			if !playersTracker.alreadyAddedPlayer(player.UserID) {
				playersTracker.addPlayer(
					playerRow{idState.nextPlayer, curGameRow.id, player.Name, player.SteamID64},
					player.UserID)
				idState.nextPlayer++
			}
		}
		nextTickId++
	})
	/*

			ticksFile, err := os.Create(localTicksCSVName)
		if err != nil {
			panic(err)
		}
		defer ticksFile.Close()
		ticksFile.WriteString(ticksHeader)

		playerAtTickFile, err := os.Create(localPlayerAtTickCSVName)
		if err != nil {
			panic(err)
		}
		defer playerAtTickFile.Close()
		playerAtTickFile.WriteString(playerAtTicksHeader)

		p.RegisterEventHandler(func(e events.FrameDone) {
			gs := p.GameState()
			if gs.IngameTick() < 1 || (ticksProcessed == 0 && gs.IngameTick() > 100000 && !gs.IsMatchStarted()) {
				return
			}
			// on the first tick save the game state
			if ticksProcessed == 0 {
				header := p.Header()
				gamesFile.WriteString(fmt.Sprintf("%d,%s,%f,%f,%s,%d\n",
					curGameID, demFilePath, (&header).FrameRate(), p.TickRate(), (&header).MapName, gameType))
				idState.nextGame++
			}
			if ticksProcessed != 0 {
				if lastInGameTick >= gs.IngameTick() {
					print("bad in game tick")
				}
			}
			lastInGameTick = gs.IngameTick()
			ticksProcessed++
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
				tickID, curRound.id, p.CurrentTime().Milliseconds(), p.CurrentFrame(), gs.IngameTick(),
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
				aimPunchAngle := player.Entity.PropertyValueMust("localdata.m_Local.m_aimPunchAngle").VectorVal
				viewPunchAngle := player.Entity.PropertyValueMust("localdata.m_Local.m_viewPunchAngle").VectorVal
				playerAtTickFile.WriteString(fmt.Sprintf(
					"%d,%d,%d,%.2f,%.2f,"+
						"%.2f,%.2f,%.2f,%.2f,%.2f,"+
						"%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"+
						"%d,%d,%d,%d,"+
						"%d,%d,%d,%d,%d,%f,%d,%d,%d,"+
						"%d,%d,%d,%d,%d,%d,%d,"+
						"%d,%d,%d,%d,%d,%d,%d,%d\n",
					playerAtTickID, getPlayerBySteamID(&playersTracker, player), tickID, player.Position().X, player.Position().Y,
					player.Position().Z, player.PositionEyes().Z, player.Velocity().X, player.Velocity().Y, player.Velocity().Z,
					player.ViewDirectionX(), player.ViewDirectionY(), aimPunchAngle.X, aimPunchAngle.Y, viewPunchAngle.X, viewPunchAngle.Y,
					side, player.Health(), player.Armor(), boolToInt(player.HasHelmet()),
					boolToInt(player.IsAlive()), boolToInt(player.IsDucking() || player.IsDuckingInProgress()), boolToInt(player.IsWalking()), boolToInt(player.IsScoped()), boolToInt(player.IsAirborne()), player.FlashDuration, activeWeapon, primaryWeapon, primaryBulletsClip,
					primaryBulletsReserve, secondaryWeapon, secondaryBulletsClip, secondaryBulletsReserve, numHE, numFlash, numSmoke,
					numIncendiary, numMolotov, numDecoy, numZeus, boolToInt(hasBomb), boolToInt(hasDefuser), player.Money(), player.Ping()))
			}
			idState.nextTick++
		})

		spottedFile, err := os.Create(localSpottedCSVName)
		if err != nil {
			panic(err)
		}
		defer spottedFile.Close()
		spottedFile.WriteString(spottedHeader)

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

		footstepFile, err := os.Create(localFootstepCSVName)
		if err != nil {
			panic(err)
		}
		defer footstepFile.Close()
		footstepFile.WriteString(footstepHeader)

		p.RegisterEventHandler(func(e events.Footstep) {
			curID := idState.nextFootstep
			idState.nextFootstep++
			footstepFile.WriteString(fmt.Sprintf("%d,%d,%d\n",
				curID, idState.nextTick, getPlayerBySteamID(&playersTracker, e.Player)))
		})

		weaponFireFile, err := os.Create(localWeaponFireCSVName)
		if err != nil {
			panic(err)
		}
		defer weaponFireFile.Close()
		weaponFireFile.WriteString(weaponFireHeader)

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
		hurtFile.WriteString(hurtHeader)

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
		killsFile.WriteString(hurtHeader)

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
		grenadesFile.WriteString(grenadeHeader)

		grenadeTrajectoriesFile, err := os.Create(localGrenadeTrajectoriesCSVName)
		if err != nil {
			panic(err)
		}
		defer grenadeTrajectoriesFile.Close()
		grenadeTrajectoriesFile.WriteString(grenadeTrajectoryHeader)

		p.RegisterEventHandler(func(e events.GrenadeProjectileThrow) {
			curID := idState.nextGrenade
			idState.nextGrenade++

			grenadesTracker[e.Projectile.WeaponInstance.UniqueID()] =
				append(grenadesTracker[e.Projectile.WeaponInstance.UniqueID()], grenadeRow{curID,
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
			if e.Grenade == nil {
				return
			}
			if _, ok := grenadesTracker[e.Grenade.UniqueID()]; !ok {
				return
			}
			curGrenade := grenadesTracker[e.Grenade.UniqueID()][0]
			curGrenade.activeTick = idState.nextTick
			grenadesTracker[e.Grenade.UniqueID()][0] = curGrenade
		})

		p.RegisterEventHandler(func(e events.SmokeExpired) {
			if e.Grenade == nil {
				return
			}
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
		playerFlashedFile.WriteString(playerFlashedHeader)

		p.RegisterEventHandler(func(e events.PlayerFlashed) {
			source := getPlayerBySteamID(&playersTracker, e.Attacker)
			target := getPlayerBySteamID(&playersTracker, e.Player)
			// this handles player flashed event firing twice
			lastFlashKey := SourceTarget{source, target}

			if oldTick, ok := lastFlash[lastFlashKey]; ok && oldTick == idState.nextTick {
				return
			}
			lastFlash[lastFlashKey] = idState.nextTick

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
		plantsFile.WriteString(plantHeader)

		p.RegisterEventHandler(func(e events.BombPlantBegin) {
			curID := idState.nextPlant
			idState.nextPlant++

			curPlant = plantRow{curID,
				idState.nextTick,
				0,
				getPlayerBySteamID(&playersTracker, e.Player),
				false,
				false,
			}
		})

		p.RegisterEventHandler(func(e events.BombPlantAborted) {
			// plants interrupted by end of round may fire twice, ignore second firing
			if curPlant.written {
				return
			}
			curPlant.endTick = idState.nextTick
			curPlant.successful = false
			plantsFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d\n",
				curPlant.id, curPlant.startTick, curPlant.endTick, curPlant.planter, boolToInt(false)))
			curPlant.written = true
		})

		p.RegisterEventHandler(func(e events.BombPlanted) {
			// plants interrupted by end of round may fire twice, ignore second firing
			if curPlant.written {
				return
			}
			curPlant.endTick = idState.nextTick
			curPlant.successful = true
			plantsFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d\n",
				curPlant.id, curPlant.startTick, curPlant.endTick, curPlant.planter, boolToInt(true)))
			curPlant.written = true
		})

		defusalsFile, err := os.Create(localDefusalsCSVName)
		if err != nil {
			panic(err)
		}
		defer defusalsFile.Close()
		defusalsFile.WriteString(defusalHeader)

		p.RegisterEventHandler(func(e events.BombDefuseStart) {
			curID := idState.nextDefusal
			idState.nextDefusal++

			curDefusal = defusalRow{curID,
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
		explosionsFile.WriteString(explosionHeader)

		p.RegisterEventHandler(func(e events.BombExplode) {
			curID := idState.nextExplosion
			idState.nextExplosion++

			explosionsFile.WriteString(fmt.Sprintf("%d,%d,%d\n", curID, curPlant.id, idState.nextTick))
		})

	*/

	err = p.ParseToEnd()
	if err != nil {
		fmt.Printf("Error in parsing. T score %d, CT score %d, progress: %f, error:\n %s\n",
			p.GameState().TeamTerrorists().Score(), p.GameState().TeamCounterTerrorists().Score(), p.Progress(), err.Error())
	}
}

func SaveStructure(idState *IDState, firstRun bool) {
	if firstRun {
		SaveEquipmentFile()
	}

	// WRITE GAME
	// create games table if it didn't exist, and append header if first run
	flags := os.O_CREATE | os.O_WRONLY
	if firstRun {
		flags = flags | os.O_TRUNC
	} else {
		flags = flags | os.O_APPEND
	}
	gamesFile, err := os.OpenFile(filepath.Join(c.TmpDir, c.GlobalGamesCSVName), flags, 0644)
	if err != nil {
		panic(err)
	}
	defer gamesFile.Close()
	if firstRun {
		gamesFile.WriteString(gamesHeader)
	}
	gamesFile.WriteString(curGameRow.toString())

	// WRITE ROUNDS
	roundsTable.saveToFile(c.LocalRoundsCSVName, roundsHeader)

	// WRITE PLAYERS
	playersTable.saveToFile(c.LocalPlayersCSVName, playersHeader)
}
