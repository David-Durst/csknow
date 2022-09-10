package internal

import (
	"bufio"
	"fmt"
	c "github.com/David-Durst/csknow/demo_parser/internal/constants"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func ParseInputStateCSV() IDState {
	idStateFile, err := os.Open(filepath.Join(c.TmpDir, c.InputStateCSVName))
	if err != nil {
		log.Fatal(err)
	}
	defer idStateFile.Close()

	var values []RowIndex
	scanner := bufio.NewScanner(idStateFile)
	for scanner.Scan() {
		// drop all the labels, save the values
		valueStr := strings.Split(scanner.Text(), ",")[1]
		i, _ := strconv.ParseInt(valueStr, 10, 64)
		values = append(values, RowIndex(i))
	}
	return IDState{values[0], values[1], values[2], values[3], values[4], values[5], values[6],
		values[7], values[8], values[9], values[10], values[11], values[12], values[13], values[14], values[15]}
}

func SaveOutputStateCSV(idState *IDState) {
	idStateFile, err := os.Create(filepath.Join(c.TmpDir, c.OutputStateCSVName))
	if err != nil {
		panic(err)
	}
	defer idStateFile.Close()
	idStateFile.WriteString(fmt.Sprintf(
		"nextGame,%d\nnextPlayer,%d\nnextRound,%d\nnextTick,%d\n"+
			"nextPlayerAtTick,%d\nnextSpotted,%d\nnextFootstep,%d\nnextWeaponFire,%d\nnextKill,%d\nnextPlayerHurt,%d\n"+
			"nextGrenade,%d\nnextGrenadeTrajectory,%d\nnextPlayerFlashed,%d\nnextPlant,%d\nnextDefusal,%d\nnextExplosion,%d\n",
		idState.nextGame, idState.nextPlayer, idState.nextRound, idState.nextTick,
		idState.nextPlayerAtTick, idState.nextSpotted, idState.nextFootstep, idState.nextWeaponFire, idState.nextKill, idState.nextPlayerHurt,
		idState.nextGrenade, idState.nextGrenadeTrajectory, idState.nextPlayerFlashed, idState.nextPlant, idState.nextDefusal, idState.nextExplosion))
}
