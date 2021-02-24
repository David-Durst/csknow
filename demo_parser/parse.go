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

func processFile(unprocessedKey string) {
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
	positionFile.WriteString("tick number,match started,game phase,rounds played,is warmup,round start,round end,round end reason,freeze time ended,t score,ct score,num players")
	for i := 0; i < 10; i++ {
		positionFile.WriteString(fmt.Sprintf(
			",player %d name,player %d team,player %d x position,player %d y position,player %d z position" +
				",player %d x view direction,player %d y view direction,player %d is alive,player %d team,player %d is blinded",
				i, i, i, i, i, i, i, i, i, i))
	}
	positionFile.WriteString(fmt.Sprintf(",demo file\n"))

	p := demoinfocs.NewParser(f)
	defer p.Close()

	// flags from other event handlers to frame done handler
	roundStart := 0
	roundEndReason := -1
	roundEnd := 0
	freezeTime := 0

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

	ticksProcessed := 0
	p.RegisterEventHandler(func(e events.FrameDone) {
		ticksProcessed++
		gs := p.GameState()
		players := getPlayers(&p)
		if len(players) != 10 {
			return
		}
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
		positionFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,", p.CurrentFrame(), matchStarted, gs.GamePhase(), gs.TotalRoundsPlayed(), isWarmup,
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
				positionFile.WriteString(fmt.Sprintf("%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d",
					players[i].Name, teamToNum(players[i].Team), players[i].Position().X, players[i].Position().Y,
					players[i].Position().Z, players[i].ViewDirectionX(), players[i].ViewDirectionY(),
					isAlive, players[i].Team, isBlinded))
			}
			positionFile.WriteString(",")
		}
		positionFile.WriteString(demFilePath + "\n")
		roundStart = 0
		roundEnd = 0
		roundEndReason = -1
	})


	spottedFile, err := os.Create(localSpottedCSVName)
	if err != nil {
		panic(err)
	}
	defer spottedFile.Close()
	spottedFile.WriteString("spotted player,")
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
	hurtFile.WriteString("shooter,weapon,tick number,demo file\n")

	p.RegisterEventHandler(func(e events.PlayerHurt) {
		if ticksProcessed < minTicks {
			return
		}

		hurtFile.WriteString(fmt.Sprintf("%s,%d,%d,%d,%d,%s,%s,%d,%s\n",
			e.Player.Name, e.ArmorDamage, e.Armor, e.HealthDamage, e.Health, e.Attacker.Name, e.Weapon.String(),
			p.CurrentFrame(), demFilePath))
	})

	grenadesFile, err := os.Create(localGrenadesCSVName)
	if err != nil {
		panic(err)
	}
	defer grenadesFile.Close()
	grenadesFile.WriteString("thrower,grenade type,tick number,demo file\n")

	p.RegisterEventHandler(func(e events.GrenadeEventIf) {
		if ticksProcessed < minTicks {
			return
		}

		grenadesFile.WriteString(fmt.Sprintf("%s,%s,%d,%s\n",
			e.Base().Thrower, e.Base().GrenadeType, p.CurrentFrame(), demFilePath))
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

		killsFile.WriteString(fmt.Sprintf("%s,%s,%s,%s,%d,%d,%d,%d,%s\n",
			e.Killer.Name, e.Victim.Name, e.Weapon.String(), e.Assister.Name,
			boolToInt(e.IsHeadshot), boolToInt(e.IsWallBang()), e.PenetratedObjects,
			p.CurrentFrame(), demFilePath))
	})

	p.ParseToEnd()
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

