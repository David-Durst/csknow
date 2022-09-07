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

func addNewRound(idState *IDState, nextTickId RowIndex, roundsTable *table[roundRow]) {
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

func ProcessStructure(unprocessedKey string, localDemName string, idState *IDState, gameType c.GameType) {
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
		addNewRound(idState, nextTickId, &unfilteredRoundsTable)
	})

	p.RegisterEventHandler(func(e events.RoundFreezetimeEnd) {
		// only update once per round, as can fire duplicate event at end of last round of match
		if unfilteredRoundsTable.tail().freezeTimeEnd == InvalidId {
			unfilteredRoundsTable.tail().freezeTimeEnd = nextTickId
		}
	})

	p.RegisterEventHandler(func(e events.RoundEnd) {
		// this is when the round objective is completed, RoundEndOfficial is when the walk around time ends
		// skip round ends on first tick, these are worthless
		if nextTickId == 0 {
			return
		}

		if unfilteredRoundsTable.len() > 0 && !unfilteredRoundsTable.tail().finished {
			unfilteredRoundsTable.tail().roundEndReason = int(e.Reason)
			unfilteredRoundsTable.tail().winner = e.Winner
			unfilteredRoundsTable.tail().endTick = nextTickId
			unfilteredRoundsTable.tail().warmup = false
		} else {
			// handle demos that start after first round start or just miss a round start event
			// assume bogus rounds are warmups
			addNewRound(idState, nextTickId, &unfilteredRoundsTable)
		}

		unfilteredRoundsTable.tail().tWins = p.GameState().Team(common.TeamTerrorists).Score()
		unfilteredRoundsTable.tail().ctWins = p.GameState().Team(common.TeamCounterTerrorists).Score()
	})

	p.RegisterEventHandler(func(e events.RoundEndOfficial) {
		// this event seems to fire the tick when you are in the next round,
		// step back 1 tick to make sure this is ending round and not the starting round
		unfilteredRoundsTable.tail().endOfficialTick = nextTickId - 1
		unfilteredRoundsTable.tail().finished = true
	})

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

	err = p.ParseToEnd()
	if err != nil {
		fmt.Printf("Error in parsing. T score %d, CT score %d, progress: %f, error:\n %s\n",
			p.GameState().TeamTerrorists().Score(), p.GameState().TeamCounterTerrorists().Score(), p.Progress(), err.Error())
	}
}

// period is regulation, OT 0, OT 1, etc
type periodIndices struct {
	lastFirstHalfStartIndex       int
	candidateFirstHalfStartIndex  int
	lastFirstHalfEndIndex         int
	lastSecondHalfStartIndex      int
	candidateSecondHalfStartIndex int
	lastSecondHalfEndIndex        int
}

var defaultIndices periodIndices = periodIndices{
	InvalidInt, InvalidInt, InvalidInt,
	InvalidInt, InvalidInt, InvalidInt}

func computePeriodIndices(lastIndices periodIndices, regulation bool, otNumber int) periodIndices {
	resultIndices := defaultIndices
	otStartScore := 0
	if !regulation {
		otStartScore = c.HalfRegulationRounds + otNumber*c.HalfOTRounds
	}
	halfPeriodRounds := c.HalfRegulationRounds
	if !regulation {
		halfPeriodRounds = c.HalfOTRounds
	}

	for i := lastIndices.lastSecondHalfEndIndex + 1; i < filteredRoundsTable.len(); i++ {
		ctPeriodWins := filteredRoundsTable.rows[i].ctWins - otStartScore
		tPeriodWins := filteredRoundsTable.rows[i].tWins - otStartScore
		if ctPeriodWins == 0 && tPeriodWins == 0 {
			resultIndices.candidateFirstHalfStartIndex = i
		}
		// end half when rounds are half total period rounds
		if ctPeriodWins+tPeriodWins+1 == halfPeriodRounds {
			resultIndices.lastFirstHalfStartIndex = resultIndices.candidateFirstHalfStartIndex
			resultIndices.lastFirstHalfEndIndex = i
		}
	}
	for i := 0; i < filteredRoundsTable.len(); i++ {
		ctPeriodWins := filteredRoundsTable.rows[i].ctWins - otStartScore
		tPeriodWins := filteredRoundsTable.rows[i].tWins - otStartScore
		if ctPeriodWins+tPeriodWins == halfPeriodRounds {
			resultIndices.candidateSecondHalfStartIndex = i
		}
		// last round where a team hasn't clinched is last round of the period
		if (ctPeriodWins == halfPeriodRounds || tPeriodWins == halfPeriodRounds) &&
			(ctPeriodWins < halfPeriodRounds || tPeriodWins < halfPeriodRounds) {
			resultIndices.lastSecondHalfStartIndex = resultIndices.candidateSecondHalfStartIndex
			resultIndices.lastSecondHalfEndIndex = i
		}
	}

	return resultIndices
}

func FilterRounds(idState *IDState) {
	filteredRoundsTable.rows = make([]roundRow, unfilteredRoundsTable.len())
	copy(filteredRoundsTable.rows, unfilteredRoundsTable.rows)
	if filteredRoundsTable.len() == 0 {
		return
	}

	// save first id to continue based on, will need it since modifying table, possibly dropping first round
	curRoundId := filteredRoundsTable.rows[0].id

	// first get rid of the easy stuff: warmup rounds
	newRoundIndex := 0
	for i := 0; i < filteredRoundsTable.len(); i++ {
		if !filteredRoundsTable.rows[i].warmup {
			filteredRoundsTable.rows[newRoundIndex] = filteredRoundsTable.rows[i]
			newRoundIndex++
		}
	}
	filteredRoundsTable.rows = filteredRoundsTable.rows[:newRoundIndex]
	var indices []periodIndices

	// next, grab the first half
	// 319_titan-epsilon_de_dust2.dem is example of warmup using sourcemod plugin
	// so not marked as warmup but is warmup
	// only way I've come up with to remove these is to look for last first half
	indices = append(indices, computePeriodIndices(defaultIndices, true, 0))

	for otNumber := 0; true; otNumber++ {
		tmpIndices := computePeriodIndices(indices[len(indices)-1], false, otNumber)
		if tmpIndices.lastSecondHalfEndIndex != InvalidInt {
			indices = append(indices, tmpIndices)
		} else {
			break
		}
	}

	// merge periods and mark OT rounds
	tmpRounds := make([]roundRow, 0)
	for i := 0; i < len(indices); i++ {
		for j := indices[i].lastFirstHalfStartIndex; j <= indices[i].lastFirstHalfEndIndex; j++ {
			if i > 0 {
				filteredRoundsTable.rows[j].overtime = true
			}
			tmpRounds = append(tmpRounds, filteredRoundsTable.rows[j])
		}
		for j := indices[i].lastSecondHalfStartIndex; j <= indices[i].lastSecondHalfEndIndex; j++ {
			if i > 0 {
				filteredRoundsTable.rows[j].overtime = true
			}
			// 2351898_128353_g2-vs-ence-m3-dust2_5f2f16a6-292a-11ec-8e27-0a58a9feac02.dem
			// end of OT has no officila round end as using plugin to manage OT, just ends as soon as round over
			// so if official round end invalid at end of period, just make it equal to regular round end
			if filteredRoundsTable.rows[j].endOfficialTick == InvalidId {
				filteredRoundsTable.rows[j].endOfficialTick = filteredRoundsTable.rows[j].endTick
			}
			tmpRounds = append(tmpRounds, filteredRoundsTable.rows[j])
		}
	}
	filteredRoundsTable.rows = tmpRounds

	// first round ids/numbers
	for i := 0; i < filteredRoundsTable.len(); i++ {
		filteredRoundsTable.rows[i].id = curRoundId
		curRoundId++
	}
	idState.nextRound = curRoundId
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
	unfilteredRoundsTable.saveToFile(c.LocalUnfilteredRoundsCSVName, roundsHeader)
	filteredRoundsTable.saveToFile(c.LocalFilteredRoundsCSVName, roundsHeader)

	// WRITE PLAYERS
	playersTable.saveToFile(c.LocalPlayersCSVName, playersHeader)
}
