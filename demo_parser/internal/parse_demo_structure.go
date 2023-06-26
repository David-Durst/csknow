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
	"strings"
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

func ProcessStructure(unprocessedKey string, localDemName string, idState *IDState, gameType c.GameType) bool {
	curGameRow.id = idState.nextGame
	// increment this locally as won't actually record ticks until after past processing sturcture
	nextTickId := idState.nextTick
	f, err := os.Open(localDemName)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	cfg := demoinfocs.DefaultParserConfig
	cfg.IgnoreErrBombsiteIndexNotFound = true
	p := demoinfocs.NewParserWithConfig(f, cfg)
	defer p.Close()

	header, err := p.ParseHeader()
	if err != nil || header.FrameRate() < 120 {
		fmt.Printf("skipping %s as frame rate is %f\n", localDemName, header.FrameRate())
		return false
	}

	p.RegisterEventHandler(func(e events.RoundStart) {
		// for now, adding all rounds, so can examine again garbage at end of match
		// initializing as warmup, will fix to non-warmup when get a round end call
		addNewRound(idState, nextTickId, &unfilteredRoundsTable)
	})

	p.RegisterEventHandler(func(e events.RoundFreezetimeEnd) {
		// if no round start yet, skip the freeze time end
		if unfilteredRoundsTable.len() == 0 {
			return
		}
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
		if unfilteredRoundsTable.len() > 0 && unfilteredRoundsTable.tail().endOfficialTick == InvalidId {
			// this event seems to fire the tick when you are in the next round,
			// step back 30 ticks to make sure this is end of current round and not starting the next round
			unfilteredRoundsTable.tail().endOfficialTick = nextTickId - 30
			if unfilteredRoundsTable.tail().endOfficialTick < unfilteredRoundsTable.tail().endTick {
				unfilteredRoundsTable.tail().endOfficialTick = unfilteredRoundsTable.tail().endTick
			}
			unfilteredRoundsTable.tail().finished = true
		} else {
			// handle demos that have a round end officila with no round end
			// assume bogus rounds are warmups
			addNewRound(idState, nextTickId, &unfilteredRoundsTable)
		}
	})

	playersTracker.init()
	if idState.nextPlayer == 0 {
		playersTable.append(playerRow{InvalidId, InvalidId, "invalid", 0})
	}

	p.RegisterEventHandler(func(e events.FrameDone) {
		players := getPlayers(&p)

		// skip ticks until at least one player is connected
		if len(players) == 0 {
			return
		}

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
		sort.Slice(players, func(i int, j int) bool {
			return players[i].Name < players[j].Name
		})
		for _, player := range players {
			if !playersTracker.alreadyAddedPlayer(player) {
				playerName := strings.ReplaceAll(player.Name, ",", "_")
				playersTracker.addPlayer(
					playerRow{idState.nextPlayer, curGameRow.id, playerName,
						player.SteamID64}, player)
				idState.nextPlayer++
			}
		}
		nextTickId++
	})

	err = p.ParseToEnd()
	if err != nil {
		fmt.Printf("Error in parsing. T score %d, CT score %d, progress: %f, error:\n %s\n",
			p.GameState().TeamTerrorists().Score(), p.GameState().TeamCounterTerrorists().Score(), p.Progress(), err.Error())
		return false
	}
	if !strings.Contains(curGameRow.mapName, "de_dust2") && !strings.Contains(curGameRow.mapName, "DE_DUST2") {
		fmt.Printf("skipping %s as map is %s\n", localDemName, curGameRow.mapName)
		return false
	}
	return true
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

func validPeriodIndices(pi periodIndices) bool {
	return pi.lastFirstHalfStartIndex != InvalidInt && pi.lastFirstHalfEndIndex != InvalidInt &&
		pi.lastSecondHalfStartIndex != InvalidInt && pi.lastSecondHalfEndIndex != InvalidInt
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
		// 2354401_134278_gambit-vs-faze-m1-dust2_faae56c6-972c-11ec-80cf-0a58a9feac02.dem - first round is invalid
		// just let that one go and grab the first round
		if ctPeriodWins == 0 && tPeriodWins == 0 || resultIndices.candidateFirstHalfStartIndex == InvalidInt {
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

func FixRounds() {
	// if round has no end, give it same end as start
	// aim_test.dem has rounds with no ends
	// if round has no official end, give it same official end as it's normal end
	for i, _ := range unfilteredRoundsTable.rows {
		if unfilteredRoundsTable.rows[i].endTick == InvalidId {
			unfilteredRoundsTable.rows[i].endTick = unfilteredRoundsTable.rows[i].startTick
		}
	}
	// 2351898_128353_g2-vs-ence-m3-dust2_5f2f16a6-292a-11ec-8e27-0a58a9feac02.dem
	// end of OT has no officila round end as using plugin to manage OT, just ends as soon as round over
	// so if official round end invalid at end of period, just make it equal to regular round end
	for i, _ := range unfilteredRoundsTable.rows {
		if unfilteredRoundsTable.rows[i].endOfficialTick == InvalidId {
			unfilteredRoundsTable.rows[i].endOfficialTick = unfilteredRoundsTable.rows[i].endTick
		}
	}
}

func FilterRounds(idState *IDState, shouldFilterRounds bool) bool {
	filteredRoundsTable.rows = make([]roundRow, unfilteredRoundsTable.len())
	copy(filteredRoundsTable.rows, unfilteredRoundsTable.rows)
	if filteredRoundsTable.len() == 0 {
		fmt.Printf("FilterRounds terminating early but successfully with no filtered rounds\n")
		return true
	}
	if !shouldFilterRounds {
		// drop empty rounds at least
		newRoundIndex := 0
		for i := 0; i < filteredRoundsTable.len(); i++ {
			if filteredRoundsTable.rows[i].startTick != filteredRoundsTable.rows[i].endTick {
				filteredRoundsTable.rows[newRoundIndex] = filteredRoundsTable.rows[i]
				newRoundIndex++
			}
		}
		filteredRoundsTable.rows = filteredRoundsTable.rows[:newRoundIndex]
		for i := 0; i < filteredRoundsTable.len(); i++ {
			filteredRoundsTable.rows[i].id = RowIndex(i)
		}
		return true
	}

	// save first id to continue based on, will need it since modifying table, possibly dropping first round
	curRoundId := filteredRoundsTable.rows[0].id

	// first get rid of the easy stuff: warmup rounds and those not won by a specific team
	newRoundIndex := 0
	for i := 0; i < filteredRoundsTable.len(); i++ {
		if !filteredRoundsTable.rows[i].warmup && (filteredRoundsTable.rows[i].winner == common.TeamTerrorists ||
			filteredRoundsTable.rows[i].winner == common.TeamCounterTerrorists) {
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

	if !validPeriodIndices(indices[0]) {
		fmt.Printf("skipping demo as no valid regulation period indices, only %d filtered rounds\n",
			filteredRoundsTable.len())
		return false
	}

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
		// 2358784_143237_honoris-vs-anonymo-m2-dust2_bc0e8b54-338d-11ed-b4ce-0a58a9feac02.dem
		// this demo drops the round end event for the end of first OT half
		// rather than recovering a rare OT period, just drop any OT period with an invalid index
		if !validPeriodIndices(indices[i]) {
			continue
		}
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
	return true
}

func FlushStructure(firstRun bool) {
	if firstRun {
		SaveEquipmentFile()
		SaveGameTypesFile()
		SaveHitGroupsFile()
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
	unfilteredRoundsTable.flush(true)
	filteredRoundsTable.flush(true)

	// WRITE PLAYERS
	playersTable.flush(true)
}
