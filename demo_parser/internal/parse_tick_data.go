package internal

import (
	"fmt"
	c "github.com/David-Durst/csknow/demo_parser/internal/constants"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs/common"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs/events"
	"os"
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
		}
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

	// https://github.com/markus-wa/demoinfocs-golang/issues/160#issuecomment-556075640 shows sequence of events
	p.RegisterEventHandler(func(e events.GrenadeProjectileThrow) {
		curID := idState.nextGrenade
		idState.nextGrenade++

		if grenadeTracker.alreadyAddedGrenade(e.Projectile.UniqueID()) {
			fmt.Printf("Adding grenade id twice %d", e.Projectile.UniqueID())
		}

		grenadeTracker.addGrenade(grenadeRow{
			curID, playersTracker.getPlayerIdFromGameData(e.Projectile.Thrower),
			e.Projectile.WeaponInstance.Type, idState.nextTick,
			InvalidId, InvalidId, InvalidId, nil,
		}, e.Projectile.UniqueID())
	})

	saveGrenade := func(id int64) {
		if !grenadeTracker.alreadyAddedGrenade(id) {
			fmt.Printf("Can't save grenade never added %d", id)
		}
		// molotovs and incendiaries are destoryed before effect ends so only save grenade trajectory once
		// both have happened
		curGrenade := grenadeTable.rows[grenadeTracker.getGrenadeIdFromGameData(id)]
		if curGrenade.destroyTick != InvalidId && curGrenade.expiredTick != InvalidId {
			for i := range curGrenade.trajectory {
				curTrajectoryID := idState.nextGrenadeTrajectory
				idState.nextGrenadeTrajectory++
				grenadeTrajectoryTable.append(grenadeTrajectoryRow{
					curTrajectoryID, curGrenade.id, i,
					curGrenade.trajectory[i].X, curGrenade.trajectory[i].Y, curGrenade.trajectory[i].Z,
				})
			}
		}
	}

	p.RegisterEventHandler(func(e events.HeExplode) {
		if !grenadeTracker.alreadyAddedGrenade(e.Grenade.UniqueID()) {
			return
		}
		curGrenade := grenadeTable.get(grenadeTracker.getGrenadeIdFromGameData(e.Grenade.UniqueID()))
		curGrenade.activeTick = idState.nextTick
		curGrenade.expiredTick = idState.nextTick
	})

	p.RegisterEventHandler(func(e events.FlashExplode) {
		if !grenadeTracker.alreadyAddedGrenade(e.Grenade.UniqueID()) {
			return
		}
		curGrenade := grenadeTable.get(grenadeTracker.getGrenadeIdFromGameData(e.Grenade.UniqueID()))
		curGrenade.activeTick = idState.nextTick
		curGrenade.expiredTick = idState.nextTick
	})

	p.RegisterEventHandler(func(e events.DecoyStart) {
		if !grenadeTracker.alreadyAddedGrenade(e.Grenade.UniqueID()) {
			return
		}
		curGrenade := grenadeTable.get(grenadeTracker.getGrenadeIdFromGameData(e.Grenade.UniqueID()))
		curGrenade.activeTick = idState.nextTick
	})

	p.RegisterEventHandler(func(e events.DecoyExpired) {
		if !grenadeTracker.alreadyAddedGrenade(e.Grenade.UniqueID()) {
			return
		}
		curGrenade := grenadeTable.get(grenadeTracker.getGrenadeIdFromGameData(e.Grenade.UniqueID()))
		curGrenade.expiredTick = idState.nextTick
	})

	p.RegisterEventHandler(func(e events.SmokeStart) {
		if !grenadeTracker.alreadyAddedGrenade(e.Grenade.UniqueID()) {
			return
		}
		curGrenade := grenadeTable.get(grenadeTracker.getGrenadeIdFromGameData(e.Grenade.UniqueID()))
		curGrenade.activeTick = idState.nextTick
	})

	p.RegisterEventHandler(func(e events.SmokeExpired) {
		if !grenadeTracker.alreadyAddedGrenade(e.Grenade.UniqueID()) {
			return
		}
		curGrenade := grenadeTable.get(grenadeTracker.getGrenadeIdFromGameData(e.Grenade.UniqueID()))
		curGrenade.expiredTick = idState.nextTick
	})

	p.RegisterEventHandler(func(e events.FireGrenadeStart) {
		if !grenadeTracker.alreadyAddedGrenade(e.Grenade.UniqueID()) {
			return
		}
		curGrenade := grenadeTable.get(grenadeTracker.getGrenadeIdFromGameData(e.Grenade.UniqueID()))
		curGrenade.activeTick = idState.nextTick
	})

	p.RegisterEventHandler(func(e events.FireGrenadeExpired) {
		if !grenadeTracker.alreadyAddedGrenade(e.Grenade.UniqueID()) {
			return
		}
		curGrenade := grenadeTable.get(grenadeTracker.getGrenadeIdFromGameData(e.Grenade.UniqueID()))
		curGrenade.expiredTick = idState.nextTick
	})

	p.RegisterEventHandler(func(e events.GrenadeProjectileDestroy) {
		if !grenadeTracker.alreadyAddedGrenade(e.Projectile.WeaponInstance.UniqueID()) {
			return
		}
		curGrenade := grenadeTable.get(grenadeTracker.getGrenadeIdFromGameData(e.Projectile.WeaponInstance.UniqueID()))
		// he grenade destroy happens when smoke from after explosion fades
		// still some smoke left, but totally visible, when smoke grenade expires
		// fire grenades are destroyed as soon as land, then burn for a while
		//fmt.Printf("destroying uid: %d\n", e.Projectile.WeaponInstance.UniqueID())
		curGrenade.destroyTick = idState.nextTick
		curGrenade.trajectory = e.Projectile.Trajectory
		saveGrenade(e.Projectile.WeaponInstance.UniqueID())
	})

	p.RegisterEventHandler(func(e events.PlayerFlashed) {
		thrower := playersTracker.getPlayerIdFromGameData(e.Attacker)
		victim := playersTracker.getPlayerIdFromGameData(e.Player)

		// this handles player flashed event firing twice
		if playerFlashedTable.len() > 0 && playerFlashedTable.tail().thrower == thrower &&
			playerFlashedTable.tail().victim == victim && playerFlashedTable.tail().tickId == idState.nextTick {
			return
		}

		curID := idState.nextPlayerFlashed
		idState.nextPlayerFlashed++
		playerFlashedTable.append(playerFlashedRow{
			curID, idState.nextTick,
			grenadeTracker.getGrenadeIdFromGameData(e.Projectile.WeaponInstance.UniqueID()),
			thrower, victim,
		})
	})

	p.RegisterEventHandler(func(e events.BombPlantBegin) {
		curID := idState.nextPlant
		idState.nextPlant++

		plantTable.append(plantRow{
			curID, idState.nextTick, InvalidId,
			playersTracker.getPlayerIdFromGameData(e.Player), false,
		})
	})

	p.RegisterEventHandler(func(e events.BombPlantAborted) {
		// plants interrupted by end of round may fire twice, this will just take latest result
		plant := plantTable.tail()
		plant.endTick = idState.nextTick
		plant.successful = false
	})

	p.RegisterEventHandler(func(e events.BombPlanted) {
		// plants interrupted by end of round may fire twice, this will just take latest result
		plant := plantTable.tail()
		plant.endTick = idState.nextTick
		plant.successful = true
	})

	p.RegisterEventHandler(func(e events.BombDefuseStart) {
		curID := idState.nextDefusal
		idState.nextDefusal++

		defusalTable.append(defusalRow{
			curID, plantTable.tail().id, idState.nextTick, InvalidId,
			playersTracker.getPlayerIdFromGameData(e.Player), false,
		})
	})

	p.RegisterEventHandler(func(e events.BombDefuseAborted) {
		defusal := defusalTable.tail()
		defusal.endTick = idState.nextTick
		defusal.successful = false
	})

	p.RegisterEventHandler(func(e events.BombDefused) {
		defusal := defusalTable.tail()
		defusal.endTick = idState.nextTick
		defusal.successful = true
	})

	p.RegisterEventHandler(func(e events.BombExplode) {
		curID := idState.nextExplosion
		idState.nextExplosion++
		explosionTable.append(explosionRow{curID, plantTable.tail().id, idState.nextTick})
	})

	err = p.ParseToEnd()
	if err != nil {
		fmt.Printf("Error in parsing. T score %d, CT score %d, progress: %f, error:\n %s\n",
			p.GameState().TeamTerrorists().Score(), p.GameState().TeamCounterTerrorists().Score(), p.Progress(), err.Error())
	}
}

func SaveTickData(idState *IDState) {
	ticksTable.saveToFile(c.LocalTicksCSVName, ticksHeader)
	playerAtTicksTable.saveToFile(c.LocalPlayerAtTickCSVName, playerAtTicksHeader)
	spottedTable.saveToFile(c.LocalSpottedCSVName, spottedHeader)
	footstepTable.saveToFile(c.LocalFootstepCSVName, footstepHeader)
	weaponFireTable.saveToFile(c.LocalWeaponFireCSVName, weaponFireHeader)
	hurtTable.saveToFile(c.LocalHurtCSVName, hurtHeader)
	killTable.saveToFile(c.LocalKillsCSVName, killHeader)
	grenadeTable.saveToFile(c.LocalGrenadesCSVName, grenadeHeader)
	grenadeTrajectoryTable.saveToFile(c.LocalGrenadeTrajectoriesCSVName, grenadeTrajectoryHeader)
	playerFlashedTable.saveToFile(c.LocalPlayerFlashedCSVName, playerFlashedHeader)
	plantTable.saveToFile(c.LocalPlantsCSVName, plantHeader)
	defusalTable.saveToFile(c.LocalDefusalsCSVName, defusalHeader)
	explosionTable.saveToFile(c.LocalExplosionsCSVName, explosionHeader)
}
