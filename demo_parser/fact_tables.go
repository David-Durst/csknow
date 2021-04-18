package main

import (
	"fmt"
	"github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/common"
	"os"
)

// function rather than make on init as need parser to init it's table
func makeEquipmentToName() map[common.EquipmentType]string {
	result := make(map[common.EquipmentType]string)
	result[common.EqUnknown] = common.EqUnknown.String()

	// Pistols
	result[common.EqP2000] = common.EqP2000.String()
	result[common.EqGlock] = common.EqGlock.String()
	result[common.EqP250] = common.EqP250.String()
	result[common.EqDeagle] = common.EqDeagle.String()
	result[common.EqFiveSeven] = common.EqFiveSeven.String()
	result[common.EqDualBerettas] = common.EqDualBerettas.String()
	result[common.EqTec9] = common.EqTec9.String()
	result[common.EqCZ] = common.EqCZ.String()
	result[common.EqUSP] = common.EqUSP.String()
	result[common.EqRevolver] = common.EqRevolver.String()

	// SMGs

	result[common.EqMP7] = common.EqMP7.String()
	result[common.EqMP9] = common.EqMP9.String()
	result[common.EqBizon] = common.EqBizon.String()
	result[common.EqMac10] = common.EqMac10.String()
	result[common.EqUMP] = common.EqUMP.String()
	result[common.EqP90] = common.EqP90.String()
	result[common.EqMP5] = common.EqMP5.String()

	// Heavy

	result[common.EqSawedOff] = common.EqSawedOff.String()
	result[common.EqNova] = common.EqNova.String()
	result[common.EqMag7] = common.EqMag7.String()
	result[common.EqSwag7] = common.EqSwag7.String()
	result[common.EqXM1014] = common.EqXM1014.String()
	result[common.EqM249] = common.EqM249.String()
	result[common.EqNegev] = common.EqNegev.String()

	// Rifles

	result[common.EqGalil] = common.EqGalil.String()
	result[common.EqFamas] = common.EqFamas.String()
	result[common.EqAK47] = common.EqAK47.String()
	result[common.EqM4A4] = common.EqM4A4.String()
	result[common.EqM4A1] = common.EqM4A1.String()
	result[common.EqScout] = common.EqScout.String()
	result[common.EqSSG08] = common.EqSSG08.String()
	result[common.EqSG556] = common.EqSG556.String()
	result[common.EqSG553] = common.EqSG553.String()
	result[common.EqAUG] = common.EqAUG.String()
	result[common.EqAWP] = common.EqAWP.String()
	result[common.EqScar20] = common.EqScar20.String()
	result[common.EqG3SG1] = common.EqG3SG1.String()

	// result[common.Equipment] = common.Equipment.String()

	result[common.EqZeus] = common.EqZeus.String()
	result[common.EqKevlar] = common.EqKevlar.String()
	result[common.EqHelmet] = common.EqHelmet.String()
	result[common.EqBomb] = common.EqBomb.String()
	result[common.EqKnife] = common.EqKnife.String()
	result[common.EqDefuseKit] = common.EqDefuseKit.String()
	result[common.EqWorld] = common.EqWorld.String()

	// Grenades

	result[common.EqDecoy] = common.EqDecoy.String()
	result[common.EqMolotov] = common.EqMolotov.String()
	result[common.EqIncendiary] = common.EqIncendiary.String()
	result[common.EqFlash] = common.EqFlash.String()
	result[common.EqSmoke] = common.EqSmoke.String()
	result[common.EqHE] = common.EqHE.String()

	return result
}

func saveEquipmentFile(equipmentToName map[common.EquipmentType]string) {
	equipmentFactTable, err := os.Create(localEquipmentFactTable)
	if err != nil {
		panic(err)
	}
	defer equipmentFactTable.Close()
	equipmentFactTable.WriteString("id,name\n")

	for id, name := range equipmentToName {
		equipmentFactTable.WriteString(fmt.Sprintf("%d,%s\n", id, name))
	}
}

func saveGameTypesFile() map[string]int {
	gameTypeFactTable, err := os.Create(localGameTypeFactTable)
	if err != nil {
		panic(err)
	}
	defer gameTypeFactTable.Close()
	gameTypeFactTable.WriteString("id,table_type\n")

	nameToID := make(map[string]int)
	nameToID["pros"] = 0
	nameToID["bots"] = 1

	for name, id := range nameToID {
		gameTypeFactTable.WriteString(fmt.Sprintf("%d,%s\n", id, name))
	}
	return nameToID
}
