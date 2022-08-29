package main

import (
	"fmt"
	"github.com/markus-wa/demoinfocs-golang/v3/pkg/demoinfocs/common"
	"os"
)

// function rather than make on init as need parser to init it's table
func saveEquipmentFile() {
	equipmentToName := make(map[common.EquipmentType]string)
	equipmentToName[-1] = "empty"
	equipmentToName[common.EqUnknown] = common.EqUnknown.String()

	// Pistols
	equipmentToName[common.EqP2000] = common.EqP2000.String()
	equipmentToName[common.EqGlock] = common.EqGlock.String()
	equipmentToName[common.EqP250] = common.EqP250.String()
	equipmentToName[common.EqDeagle] = common.EqDeagle.String()
	equipmentToName[common.EqFiveSeven] = common.EqFiveSeven.String()
	equipmentToName[common.EqDualBerettas] = common.EqDualBerettas.String()
	equipmentToName[common.EqTec9] = common.EqTec9.String()
	equipmentToName[common.EqCZ] = common.EqCZ.String()
	equipmentToName[common.EqUSP] = common.EqUSP.String()
	equipmentToName[common.EqRevolver] = common.EqRevolver.String()

	// SMGs

	equipmentToName[common.EqMP7] = common.EqMP7.String()
	equipmentToName[common.EqMP9] = common.EqMP9.String()
	equipmentToName[common.EqBizon] = common.EqBizon.String()
	equipmentToName[common.EqMac10] = common.EqMac10.String()
	equipmentToName[common.EqUMP] = common.EqUMP.String()
	equipmentToName[common.EqP90] = common.EqP90.String()
	equipmentToName[common.EqMP5] = common.EqMP5.String()

	// Heavy

	equipmentToName[common.EqSawedOff] = common.EqSawedOff.String()
	equipmentToName[common.EqNova] = common.EqNova.String()
	equipmentToName[common.EqMag7] = common.EqMag7.String()
	equipmentToName[common.EqSwag7] = common.EqSwag7.String()
	equipmentToName[common.EqXM1014] = common.EqXM1014.String()
	equipmentToName[common.EqM249] = common.EqM249.String()
	equipmentToName[common.EqNegev] = common.EqNegev.String()

	// Rifles

	equipmentToName[common.EqGalil] = common.EqGalil.String()
	equipmentToName[common.EqFamas] = common.EqFamas.String()
	equipmentToName[common.EqAK47] = common.EqAK47.String()
	equipmentToName[common.EqM4A4] = common.EqM4A4.String()
	equipmentToName[common.EqM4A1] = common.EqM4A1.String()
	equipmentToName[common.EqScout] = common.EqScout.String()
	equipmentToName[common.EqSSG08] = common.EqSSG08.String()
	equipmentToName[common.EqSG556] = common.EqSG556.String()
	equipmentToName[common.EqSG553] = common.EqSG553.String()
	equipmentToName[common.EqAUG] = common.EqAUG.String()
	equipmentToName[common.EqAWP] = common.EqAWP.String()
	equipmentToName[common.EqScar20] = common.EqScar20.String()
	equipmentToName[common.EqG3SG1] = common.EqG3SG1.String()

	// equipmentToName[common.Equipment] = common.Equipment.String()

	equipmentToName[common.EqZeus] = common.EqZeus.String()
	equipmentToName[common.EqKevlar] = common.EqKevlar.String()
	equipmentToName[common.EqHelmet] = common.EqHelmet.String()
	equipmentToName[common.EqBomb] = common.EqBomb.String()
	equipmentToName[common.EqKnife] = common.EqKnife.String()
	equipmentToName[common.EqDefuseKit] = common.EqDefuseKit.String()
	equipmentToName[common.EqWorld] = common.EqWorld.String()

	// Grenades

	equipmentToName[common.EqDecoy] = common.EqDecoy.String()
	equipmentToName[common.EqMolotov] = common.EqMolotov.String()
	equipmentToName[common.EqIncendiary] = common.EqIncendiary.String()
	equipmentToName[common.EqFlash] = common.EqFlash.String()
	equipmentToName[common.EqSmoke] = common.EqSmoke.String()
	equipmentToName[common.EqHE] = common.EqHE.String()

	equipmentFactTable, err := os.Create(localEquipmentDimTable)
	if err != nil {
		panic(err)
	}
	defer equipmentFactTable.Close()
	equipmentFactTable.WriteString("id,name\n")

	for id, name := range equipmentToName {
		equipmentFactTable.WriteString(fmt.Sprintf("%d,%s\n", id, name))
	}
}

func saveGameTypesFile() {
	gameTypeFactTable, err := os.Create(localGameTypeDimTable)
	if err != nil {
		panic(err)
	}
	defer gameTypeFactTable.Close()
	gameTypeFactTable.WriteString("id,table_type\n")

	for id, name := range gameTypes {
		gameTypeFactTable.WriteString(fmt.Sprintf("%d,%s\n", id, name))
	}
}

func saveHitGroupsFile() {
	hitGroupFactTable, err := os.Create(localHitGroupDimTable)
	if err != nil {
		panic(err)
	}
	defer hitGroupFactTable.Close()
	hitGroupFactTable.WriteString("id,table_type\n")

	groupToID := make(map[string]int)
	groupToID["HitGroupInvalid"] = -1
	groupToID["HitGroupGeneric"] = 0
	groupToID["HitGroupHead"] = 1
	groupToID["HitGroupChest"] = 2
	groupToID["HitGroupStomach"] = 3
	groupToID["HitGroupLeftArm"] = 4
	groupToID["HitGroupRightArm"] = 5
	groupToID["HitGroupLeftLeg"] = 6
	groupToID["HitGroupRightLeg"] = 7
	groupToID["HitGroupGear"] = 10

	for name, id := range groupToID {
		hitGroupFactTable.WriteString(fmt.Sprintf("%d,%s\n", id, name))
	}
}
