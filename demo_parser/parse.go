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

func processFile(unprocessedKey string) {
	demFilePath := path.Base(unprocessedKey)
	f, err := os.Open(localDemName)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	csvFile, err := os.Create(localCSVName)
	if err != nil {
		panic(err)
	}
	defer csvFile.Close()

	p := demoinfocs.NewParser(f)
	defer p.Close()

	// Register handler on kill events
	csvFile.WriteString("tick number,match started,game phase,rounds played,is warmup,round start,round end,round end reason,freeze time ended,t score,ct score,num players")
	for i := 0; i < 10; i++ {
		csvFile.WriteString(fmt.Sprintf(",player %d name,player %d team,player %d x postion,player %d y position,player %d z position,player %d x view direction,player %d y view direction", i, i, i, i, i, i, i))
	}
	csvFile.WriteString(fmt.Sprintf(",demo file key\n"))
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
		players := gs.Participants().Playing()
		if len(players) != 10 {
			return
		}
		// skip the first couple seconds as tick number  can have issues with these
		// this should be fine as should just be warmup, which is 3 seconds despite fact I told cs to disable
		if ticksProcessed < 30 {
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
		csvFile.WriteString(fmt.Sprintf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,", p.CurrentFrame(), matchStarted, gs.GamePhase(), gs.TotalRoundsPlayed(), isWarmup,
			roundStart, roundEnd, roundEndReason, freezeTime, gs.TeamTerrorists().Score(), gs.TeamCounterTerrorists().Score(), len(players)))
		sort.Slice(players, func(i int, j int) bool {
			return players[i].Name < players[j].Name
		})
		for i := 0; i < 10; i++ {
			if i >= len(players) {
				csvFile.WriteString(",,,,,")
			} else {
				csvFile.WriteString(fmt.Sprintf("%s,%d,%.2f,%.2f,%.2f,%.2f,%.2f", players[i].Name, teamToNum(players[i].Team),
					players[i].Position().X, players[i].Position().Y, players[i].Position().Z,
					players[i].ViewDirectionX(), players[i].ViewDirectionY()))
			}
			csvFile.WriteString(",")
		}
		csvFile.WriteString(demFilePath + "\n")
		roundStart = 0
		roundEnd = 0
		roundEndReason = -1
	})
	p.ParseToEnd()
	if err != nil {
		panic(err)
	}
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


