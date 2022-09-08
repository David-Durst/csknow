package internal

import (
	"fmt"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs/common"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs/events"
	"os"
	"strconv"
)

func getPlayers(p *demoinfocs.Parser) []*common.Player {
	return (*p).GameState().Participants().Playing()
}

func ProcessTickData(unprocessedKey string, localDemName string, idState *IDState) {
	fmt.Printf("localDemName: %s\n", localDemName)
	f, err := os.Open(localDemName)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	p := demoinfocs.NewParser(f)
	defer p.Close()

	ticksProcessed := 0
	p.RegisterEventHandler(func(e events.FrameDone) {
		gs := p.GameState()
		if ticksProcessed != 0 {
			if ticksTable.tail().gameTickNumber >= gs.IngameTick() {
				print("bad in game tick")
			}
		}

		ticksProcessed++

		curRound := InvalidId
		for _, el := range filteredRoundsTable.rows {
			if idState.nextTick >= el.startTick && idState.nextTick <= el.endOfficialTick {
				curRound = el.id
				break
			}
		}

		tickID := idState.nextTick
		ticksTable.append(tickRow{
			idState.nextTick, curRound, p.CurrentTime().Milliseconds(),
			p.CurrentFrame(), gs.IngameTick(),
			playersTracker.getPlayerIdFromGameData(gs.Bomb().Carrier),
			gs.Bomb().Position().X, gs.Bomb().Position().Y, gs.Bomb().Position().Z,
		})

		players := getPlayers(&p)
		for _, player := range players {
			playerAtTickID := idState.nextPlayerAtTick
			idState.nextPlayerAtTick++
			primaryWeapon := common.EqUnknown
			primaryBulletsClip := 0
			primaryBulletsReserve := 0
			secondaryWeapon := common.EqUnknown
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
					secondaryWeapon = weapon.Type
					secondaryBulletsClip = weapon.AmmoInMagazine()
					secondaryBulletsReserve = weapon.AmmoReserve()
				} else if weapon.Class() == common.EqClassSMG || weapon.Class() == common.EqClassHeavy ||
					weapon.Class() == common.EqClassRifle {
					primaryWeapon = weapon.Type
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
			activeWeapon := common.EqUnknown
			if player.ActiveWeapon() != nil {
				activeWeapon = player.ActiveWeapon().Type
			}
			side := spectator
			if player.Team == common.TeamCounterTerrorists {
				side = ctSide
			} else if player.Team == common.TeamTerrorists {
				side = tSide
			}
			aimPunchAngle := player.Entity.PropertyValueMust("localdata.m_Local.m_aimPunchAngle").VectorVal
			viewPunchAngle := player.Entity.PropertyValueMust("localdata.m_Local.m_viewPunchAngle").VectorVal
			duckAmount := player.Entity.PropertyValueMust("m_flDuckAmount").FloatVal
			playerAtTicksTable.append(playerAtTickRow{
				playerAtTickID, playersTracker.getPlayerIdFromGameData(player), tickID,
				player.Position().X, player.Position().Y, player.Position().Z, player.PositionEyes().Z,
				player.Velocity().X, player.Velocity().Y, player.Velocity().Z,
				player.ViewDirectionX(), player.ViewDirectionY(),
				aimPunchAngle.X, aimPunchAngle.Y, viewPunchAngle.X, viewPunchAngle.Y,
				side, player.Health(), player.Armor(), player.HasHelmet(),
				player.IsAlive(), player.Flags().DuckingKeyPressed(), duckAmount,
				player.IsWalking(), player.IsScoped(), player.IsAirborne(),
				player.FlashDuration, activeWeapon,
				primaryWeapon, primaryBulletsClip, primaryBulletsReserve,
				secondaryWeapon, secondaryBulletsClip, secondaryBulletsReserve,
				numHE, numFlash, numSmoke, numIncendiary, numMolotov,
				numDecoy, numZeus, hasBomb, hasDefuser, player.Money(), player.Ping(),
			})
		}
		idState.nextTick++
	})

	p.RegisterEventHandler(func(e events.PlayerSpottersChanged) {
		players := getPlayers(&p)
		for _, possibleSpotter := range players {
			curID := idState.nextSpotted
			idState.nextSpotted++
			spottedTable.append(spottedRow{
				curID, idState.nextTick, playersTracker.getPlayerIdFromGameData(e.Spotted),
				playersTracker.getPlayerIdFromGameData(possibleSpotter),
				e.Spotted.IsSpottedBy(possibleSpotter),
		})
	})

	p.RegisterEventHandler(func(e events.Footstep) {
		curID := idState.nextFootstep
		idState.nextFootstep++
		footstepTable.append(footstepRow{
			curID, idState.nextTick, playersTracker.getPlayerIdFromGameData(e.Player),
		})
	})

	p.RegisterEventHandler(func(e events.WeaponFire) {
		curID := idState.nextWeaponFire
		idState.nextWeaponFire++
		weaponFireTable.append(weaponFireRow{
			curID, idState.nextTick,
			playersTracker.getPlayerIdFromGameData(e.Shooter), e.Weapon.Type,
		})
	})

	p.RegisterEventHandler(func(e events.PlayerHurt) {
		curID := idState.nextPlayerHurt
		idState.nextPlayerHurt++
		hurtTable.append(hurtRow{
			curID, idState.nextTick, playersTracker.getPlayerIdFromGameData(e.Player),
			playersTracker.getPlayerIdFromGameData(e.Attacker),
			e.Weapon.Type, e.ArmorDamage, e.Armor,
			e.HealthDamage, e.Health, e.HitGroup,
		})
	})

	p.RegisterEventHandler(func(e events.Kill) {
		curID := idState.nextKill
		idState.nextKill++
		killTable.append(killRow{
			curID, idState.nextTick, playersTracker.getPlayerIdFromGameData(e.Killer),
			playersTracker.getPlayerIdFromGameData(e.Victim), e.Weapon.Type,
			playersTracker.getPlayerIdFromGameData(e.Assister), e.IsHeadshot, e.IsWallBang(),
			e.PenetratedObjects,
		})
	})

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

	err = p.ParseToEnd()
	if err != nil {
		fmt.Printf("Error in parsing. T score %d, CT score %d, progress: %f, error:\n %s\n",
			p.GameState().TeamTerrorists().Score(), p.GameState().TeamCounterTerrorists().Score(), p.Progress(), err.Error())
	}
}
