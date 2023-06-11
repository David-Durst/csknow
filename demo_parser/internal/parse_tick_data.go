package internal

import (
	"fmt"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs/common"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs/events"
	"os"
	"strings"
)

func getPlayers(p *demoinfocs.Parser) []*common.Player {
	return (*p).GameState().Participants().Playing()
}

func ProcessTickData(localDemName string, idState *IDState) {
	f, err := os.Open(localDemName)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	cfg := demoinfocs.DefaultParserConfig
	cfg.IgnoreErrBombsiteIndexNotFound = true
	p := demoinfocs.NewParserWithConfig(f, cfg)
	defer p.Close()

	p.RegisterEventHandler(func(e events.FrameDone) {
		gs := p.GameState()
		players := getPlayers(&p)

		// skip ticks until at least one player is connected
		if len(players) == 0 {
			return
		}

		if ticksTable.len() > 0 && ticksTable.tail().gameTickNumber >= gs.IngameTick() {
			fmt.Printf("bad in game tick: id %d, demo tick %d, in-game tick %d\n",
				idState.nextTick, p.CurrentFrame(), gs.IngameTick())
		}

		curRound := InvalidId
		for _, el := range filteredRoundsTable.rows {
			if idState.nextTick >= el.startTick && idState.nextTick <= el.endOfficialTick {
				curRound = el.id
				break
			}
		}

		tickID := idState.nextTick
		carrierId := InvalidId
		if gs.Bomb().Carrier != nil {
			carrierId = playersTracker.getPlayerIdFromGameData(gs.Bomb().Carrier)
		}

		// c4 position can be wrong if teleport c4. get latest c4 pos on teleport
		c4Pos := gs.Bomb().Position()
		for _, entity := range gs.Entities() {
			if entity.ServerClass().Name() == "CPlantedC4" {
				if gs.Bomb().Carrier != nil {
					println("planted c4 and carrier at same time")
				}
				c4Pos = entity.Position()
			}
		}

		ticksTable.append(tickRow{
			idState.nextTick, curRound, p.CurrentTime().Milliseconds(),
			p.CurrentFrame(), gs.IngameTick(), carrierId,
			c4Pos.X, c4Pos.Y, c4Pos.Z,
		})

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
			recoilIndex := InvalidFloat
			nextPrimaryAttack := InvalidFloat
			nextSecondaryAttack := InvalidFloat
			if player.ActiveWeapon() != nil {
				activeWeapon = player.ActiveWeapon().Type
				recoilIndex = player.ActiveWeapon().Entity.PropertyValueMust("m_flRecoilIndex").FloatVal
				nextPrimaryAttack = player.ActiveWeapon().Entity.PropertyValueMust("LocalActiveWeaponData.m_flNextPrimaryAttack").FloatVal
				nextSecondaryAttack = player.ActiveWeapon().Entity.PropertyValueMust("LocalActiveWeaponData.m_flNextSecondaryAttack").FloatVal
			}
			/*
				for _, prop := range player.Entity.Properties() {
					print(prop.Name() + "\n")
				}
			*/
			gameTime := float64(player.Entity.PropertyValueMust("localdata.m_nTickBase").IntVal) * (1. / p.TickRate())
			aimPunchAngle := player.Entity.PropertyValueMust("localdata.m_Local.m_aimPunchAngle").VectorVal
			viewPunchAngle := player.Entity.PropertyValueMust("localdata.m_Local.m_viewPunchAngle").VectorVal
			// old demos (like 319_titan-epsilon_de_dust2.dem) don't track duck amount (probably not networked at that
			// time), so just put invalid value, database should assume ducking is instantaneous for those cases
			duckAmount := InvalidFloat
			if duckAmountProperty, ok := player.Entity.PropertyValue("m_flDuckAmount"); ok {
				duckAmount = duckAmountProperty.FloatVal
			}
			playerAtTicksTable.append(playerAtTickRow{
				playerAtTickID, playersTracker.getPlayerIdFromGameData(player), tickID,
				player.Position().X, player.Position().Y, player.Position().Z, player.PositionEyes().Z,
				player.Velocity().X, player.Velocity().Y, player.Velocity().Z,
				player.ViewDirectionX(), player.ViewDirectionY(),
				// flipping so match demo parser's assigner of yaw to x and pitch to y
				aimPunchAngle.Y, aimPunchAngle.X, viewPunchAngle.Y, viewPunchAngle.X,
				recoilIndex, nextPrimaryAttack, nextSecondaryAttack, gameTime,
				int(player.Team), player.Health(), player.Armor(), player.HasHelmet(),
				player.IsAlive(), player.Flags().DuckingKeyPressed(), duckAmount,
				player.IsReloading, player.IsWalking(), player.IsScoped(), player.IsAirborne(),
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
		spottedPlayer := playersTracker.getPlayerIdFromGameData(e.Spotted)

		// victim may be invalid if they are in the server but spectating
		// I dont track these players
		if spottedPlayer == InvalidId {
			return
		}

		for _, possibleSpotter := range players {
			curID := idState.nextSpotted
			idState.nextSpotted++
			spottedTable.append(spottedRow{
				curID, idState.nextTick, spottedPlayer,
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

		gs := p.GameState()
		gs.IngameTick()
		/*
			TODO: reenable when unique ids are working better
			if grenadeTracker.alreadyAddedGrenade(e.Projectile.WeaponInstance) {
				fmt.Printf("Adding grenade id twice %d", e.Projectile.WeaponInstance.UniqueID2())
			}
		*/

		grenadeTracker.addGrenade(grenadeRow{
			curID, playersTracker.getPlayerIdFromGameData(e.Projectile.Thrower),
			e.Projectile.WeaponInstance.Type, idState.nextTick,
			InvalidId, InvalidId, InvalidId,
		}, e.Projectile.WeaponInstance)
	})

	p.RegisterEventHandler(func(e events.HeExplode) {
		if !grenadeTracker.alreadyAddedGrenade(e.Grenade) {
			return
		}
		curGrenade := grenadeTable.get(grenadeTracker.getGrenadeIdFromGameData(e.Grenade))
		curGrenade.activeTick = idState.nextTick
		curGrenade.expiredTick = idState.nextTick
	})

	p.RegisterEventHandler(func(e events.FlashExplode) {
		if !grenadeTracker.alreadyAddedGrenade(e.Grenade) {
			return
		}
		curGrenade := grenadeTable.get(grenadeTracker.getGrenadeIdFromGameData(e.Grenade))
		curGrenade.activeTick = idState.nextTick
		curGrenade.expiredTick = idState.nextTick
	})

	p.RegisterEventHandler(func(e events.DecoyStart) {
		if !grenadeTracker.alreadyAddedGrenade(e.Grenade) {
			return
		}
		curGrenade := grenadeTable.get(grenadeTracker.getGrenadeIdFromGameData(e.Grenade))
		curGrenade.activeTick = idState.nextTick
	})

	p.RegisterEventHandler(func(e events.DecoyExpired) {
		if !grenadeTracker.alreadyAddedGrenade(e.Grenade) {
			return
		}
		curGrenade := grenadeTable.get(grenadeTracker.getGrenadeIdFromGameData(e.Grenade))
		curGrenade.expiredTick = idState.nextTick
	})

	p.RegisterEventHandler(func(e events.SmokeStart) {
		if !grenadeTracker.alreadyAddedGrenade(e.Grenade) {
			return
		}
		curGrenade := grenadeTable.get(grenadeTracker.getGrenadeIdFromGameData(e.Grenade))
		curGrenade.activeTick = idState.nextTick
	})

	p.RegisterEventHandler(func(e events.SmokeExpired) {
		if !grenadeTracker.alreadyAddedGrenade(e.Grenade) {
			return
		}
		curGrenade := grenadeTable.get(grenadeTracker.getGrenadeIdFromGameData(e.Grenade))
		curGrenade.expiredTick = idState.nextTick
	})

	p.RegisterEventHandler(func(e events.FireGrenadeStart) {
		if !grenadeTracker.alreadyAddedGrenade(e.Grenade) {
			return
		}
		curGrenade := grenadeTable.get(grenadeTracker.getGrenadeIdFromGameData(e.Grenade))
		curGrenade.activeTick = idState.nextTick
	})

	p.RegisterEventHandler(func(e events.FireGrenadeExpired) {
		if !grenadeTracker.alreadyAddedGrenade(e.Grenade) {
			return
		}
		curGrenade := grenadeTable.get(grenadeTracker.getGrenadeIdFromGameData(e.Grenade))
		curGrenade.expiredTick = idState.nextTick
	})

	p.RegisterEventHandler(func(e events.GrenadeProjectileDestroy) {
		if !grenadeTracker.alreadyAddedGrenade(e.Projectile.WeaponInstance) {
			return
		}
		curGrenade := grenadeTable.get(grenadeTracker.getGrenadeIdFromGameData(e.Projectile.WeaponInstance))
		// he grenade destroy happens when smoke from after explosion fades
		// still some smoke left, but totally visible, when smoke grenade expires
		// fire grenades are destroyed as soon as land, then burn for a while
		//fmt.Printf("destroying uid: %d\n", e.Projectile.WeaponInstance.UniqueID())
		curGrenade.destroyTick = idState.nextTick
		// ok to do this just in destroy instead of also in expired like before
		// because destroy is end of projectile, expired is about fire and not projectile creating fire
		for i, el := range e.Projectile.Trajectory {
			curTrajectoryID := idState.nextGrenadeTrajectory
			idState.nextGrenadeTrajectory++
			grenadeTrajectoryTable.append(grenadeTrajectoryRow{
				curTrajectoryID, curGrenade.id, i,
				el.X, el.Y, el.Z,
			})
		}
	})

	p.RegisterEventHandler(func(e events.PlayerFlashed) {
		thrower := playersTracker.getPlayerIdFromGameData(e.Attacker)
		victim := playersTracker.getPlayerIdFromGameData(e.Player)

		// victim may be invalid if they are in the server but spectating
		// I dont track these players
		if thrower == InvalidId || victim == InvalidId {
			return
		}

		// this handles player flashed event firing twice
		if playerFlashedTable.len() > 0 && playerFlashedTable.tail().thrower == thrower &&
			playerFlashedTable.tail().victim == victim && playerFlashedTable.tail().tickId == idState.nextTick {
			return
		}

		curID := idState.nextPlayerFlashed
		idState.nextPlayerFlashed++
		playerFlashedTable.append(playerFlashedRow{
			curID, idState.nextTick,
			grenadeTracker.getGrenadeIdFromGameData(e.Projectile.WeaponInstance),
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
		// defuse mode has successful plants that never begin, add a plant row when those happen
		if plantTable.len() == 0 || plantTable.tail().endTick != InvalidId {
			curID := idState.nextPlant
			idState.nextPlant++

			plantTable.append(plantRow{
				curID, idState.nextTick, InvalidId,
				playersTracker.getPlayerIdFromGameData(e.Player), false,
			})
		}
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
		// demos/bot_retakes_data/unprocessed/bots/r_1_auto0-20230430-131945-747721314-de_dust2-Counter-Strike__Global_Offensive_c90ec300-e6ed-11ed-a337-0242ac110002.dem
		// had a defsual with no defuse start
		if defusalTable.len() == 0 {
			curID := idState.nextDefusal
			idState.nextDefusal++

			defusalTable.append(defusalRow{
				curID, plantTable.tail().id, idState.nextTick, InvalidId,
				playersTracker.getPlayerIdFromGameData(e.Player), false,
			})
		}
		defusal := defusalTable.tail()
		defusal.endTick = idState.nextTick
		defusal.successful = false
	})

	p.RegisterEventHandler(func(e events.BombDefused) {
		// demos/bot_retakes_data/unprocessed/bots/r_1_auto0-20230430-131945-747721314-de_dust2-Counter-Strike__Global_Offensive_c90ec300-e6ed-11ed-a337-0242ac110002.dem
		// had a defsual with no defuse start
		if defusalTable.len() == 0 {
			curID := idState.nextDefusal
			idState.nextDefusal++

			defusalTable.append(defusalRow{
				curID, plantTable.tail().id, idState.nextTick, InvalidId,
				playersTracker.getPlayerIdFromGameData(e.Player), false,
			})
		}
		defusal := defusalTable.tail()
		defusal.endTick = idState.nextTick
		defusal.successful = true
	})

	p.RegisterEventHandler(func(e events.BombExplode) {
		curID := idState.nextExplosion
		idState.nextExplosion++
		explosionTable.append(explosionRow{curID, plantTable.tail().id, idState.nextTick})
	})

	p.RegisterEventHandler(func(e events.SayText) {
		curID := idState.nextSay
		idState.nextSay++
		strippedNewlineText := strings.ReplaceAll(e.Text, "\n", "_")
		strippedCommaText := strings.ReplaceAll(strippedNewlineText, ",", "_")
		sayTable.append(sayRow{curID, idState.nextTick, strippedCommaText})
	})

	p.RegisterEventHandler(func(e events.RoundEndOfficial) {
		FlushTickData(false)
	})

	err = p.ParseToEnd()
	if err != nil {
		fmt.Printf("Error in parsing. T score %d, CT score %d, progress: %f, error:\n %s\n",
			p.GameState().TeamTerrorists().Score(), p.GameState().TeamCounterTerrorists().Score(), p.Progress(), err.Error())
	}
}

func FlushTickData(close bool) {
	ticksTable.flush(close)
	playerAtTicksTable.flush(close)
	spottedTable.flush(close)
	footstepTable.flush(close)
	weaponFireTable.flush(close)
	hurtTable.flush(close)
	killTable.flush(close)
	grenadeTable.flush(close)
	grenadeTrajectoryTable.flush(close)
	playerFlashedTable.flush(close)
	plantTable.flush(close)
	defusalTable.flush(close)
	explosionTable.flush(close)
	sayTable.flush(close)
}
