package main

import (
	"bufio"
	"flag"
	"fmt"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

var localDemName string
const baseStateCSVName = "global_id_state.csv"
const inputStateCSVName = "input_" + baseStateCSVName
const outputStateCSVName = "output_" + baseStateCSVName
const gamesCSVName = "global_games.csv"
const localRoundsCSVName = "local_rounds.csv"
const localPlayersCSVName = "local_players.csv"
const localTicksCSVName = "local_ticks.csv"
const localPlayerAtTickCSVName = "local_player_at_tick.csv"
const localSpottedCSVName = "local_spotted.csv"
const localWeaponFireCSVName = "local_weapon_fire.csv"
const localHurtCSVName = "local_hurt.csv"
const localGrenadesCSVName = "local_grenades.csv"
const localGrenadeTrajectoriesCSVName = "local_grenade_trajectories.csv"
const localPlayerFlashedCSVName = "local_flashed.csv"
const localPlantsCSVName = "local_plants.csv"
const localDefusalsCSVName = "local_defusals.csv"
const localExplosionsCSVName = "local_explosions.csv"
const localKillsCSVName = "local_kills.csv"
const localEquipmentDimTable = "dimension_table_equipment.csv"
const localGameTypeDimTable = "dimension_table_game_types.csv"
const localHitGroupDimTable = "dimension_table_hit_groups.csv"
const unprocessedPrefix = "demos/unprocessed2/"
const processedPrefix = "demos/processed2/"
const processedSmallPrefix = "demos/processed2_small/"
const csvPrefixBase = "demos/csvs3/"
const csvPrefixLocal = csvPrefixBase + "local/"
const csvPrefixGlobal = csvPrefixBase +  "global/"
var gameTypes = []string{"pros","bots"}
const bucketName = "csknow"

func parseInputStateCSV() IDState {
	idStateFile, err := os.Open(inputStateCSVName)
	if err != nil {
		log.Fatal(err)
	}
	defer idStateFile.Close()

	var values []int64
	scanner := bufio.NewScanner(idStateFile)
	for scanner.Scan() {
		// drop all the labels, save the values
		valueStr := strings.Split(scanner.Text(), ",")[1]
		i, _ := strconv.ParseInt(valueStr, 10, 64)
		values = append(values, i)
	}
	return IDState{values[0], values[1], values[2], values[3], values[4], values[5], values[6],
		values[7], values[8], values[9], values[10], values[11], values[12], values[13], values[14]}
}

func saveOutputStateCSV(idState *IDState) {
	idStateFile, err := os.Create(outputStateCSVName)
	if err != nil {
		panic(err)
	}
	defer idStateFile.Close()
	idStateFile.WriteString(fmt.Sprintf(
		"nextGame,%d\nnextPlayer,%d\nnextRound,%d\nnextTick,%d\n" +
			"nextPlayerAtTick,%d\nnextSpotted,%d\nnextWeaponFire,%d\nnextKill,%d\nnextPlayerHurt,%d\n" +
			"nextGrenade,%d\nnextGrenadeTrajectory,%d\nnextPlayerFlashed,%d\nnextPlant,%d\nnextDefusal,%d\nnextExplosion,%d\n",
		idState.nextGame, idState.nextPlayer, idState.nextRound, idState.nextTick,
		idState.nextPlayerAtTick, idState.nextSpotted, idState.nextWeaponFire, idState.nextKill, idState.nextPlayerHurt,
		idState.nextGrenade, idState.nextGrenadeTrajectory, idState.nextPlayerFlashed, idState.nextPlant, idState.nextDefusal, idState.nextExplosion))
}

func main() {
	startIDState := IDState{0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 0}

	// if reprocessing, don't move the demos
	firstRunPtr := flag.Bool("f", true, "set if first file processed is first overall")
	reprocessFlag := flag.Bool("r", false, "set for reprocessing demos")
	subsetReprocessFlag := flag.Bool("s", false, "set for reprocessing a subset of demos")
	// if running locally, skip the aws stuff and just return
	localFlag := flag.Bool("l", false, "set for non-aws (aka local) runs")
	localDemName := flag.String("n", "local.dem", "set for demo's name on local file system")
	flag.Parse()
	firstRun := *firstRunPtr
	if *reprocessFlag && *subsetReprocessFlag {
		fmt.Printf("-s (reprocess subset) and -r (reprocess all) can't be set at same time\n")
		os.Exit(0)
	}

	saveGameTypesFile()
	saveHitGroupsFile()
	if *localFlag {
		if !firstRun {
			startIDState = parseInputStateCSV()
		}
		processFile(*localDemName, *localDemName, &startIDState, firstRun, 1)
		saveOutputStateCSV(&startIDState)
		os.Exit(0)
	}


	sess := session.Must(session.NewSession(&aws.Config{
		Region:      aws.String("us-east-1")},
		))

	svc := s3.New(sess)

	downloader := s3manager.NewDownloader(sess)
	uploader := s3manager.NewUploader(sess)

	var filesToMove []string
	sourcePrefix := unprocessedPrefix
	if *reprocessFlag {
		sourcePrefix = processedPrefix
	} else if *subsetReprocessFlag {
		sourcePrefix = processedSmallPrefix
	}

	idStateAWS := csvPrefixBase + baseStateCSVName
	result, err := svc.ListObjectsV2(&s3.ListObjectsV2Input{
		Bucket: aws.String(bucketName),
		Prefix: &idStateAWS,
	})
	if err != nil {
		panic(err)
	}

	// if not reprocessing and already have an id state, start from there
	if *result.KeyCount == 1 && !*reprocessFlag && !*subsetReprocessFlag {
		downloadFile(downloader, *result.Contents[0].Key, inputStateCSVName)
		startIDState = parseInputStateCSV()
	}

	i := 0
	localGameTypes := gameTypes
	if *reprocessFlag || *subsetReprocessFlag {
		localGameTypes = []string{""}
	}
	for gameTypeIndex, gameTypeString := range localGameTypes {
		sourcePrefixWithType := sourcePrefix + gameTypeString + "/"
		if *reprocessFlag || *subsetReprocessFlag {
			sourcePrefixWithType = sourcePrefix
		}
		svc.ListObjectsV2Pages(&s3.ListObjectsV2Input{
			Bucket: aws.String(bucketName),
			Prefix: aws.String(sourcePrefixWithType),
		}, func(p *s3.ListObjectsV2Output, last bool) bool {
			fmt.Printf("Processing page %d\n", i)

			for _, obj := range p.Contents {
				if !strings.HasSuffix(*obj.Key, ".dem") {
					fmt.Printf("Skipping: %s\n", *obj.Key)
					continue
				}
				fmt.Printf("Handling file: %s\n", *obj.Key)
				downloadFile(downloader, *obj.Key, *localDemName)
				processFile(*obj.Key, *localDemName, &startIDState, firstRun, gameTypeIndex)
				firstRun = false
				uploadCSVs(uploader, *obj.Key)
				filesToMove = append(filesToMove, *obj.Key)
			}

			return true
		})
	}
	uploadFile(uploader, gamesCSVName, "global_games", csvPrefixGlobal)
	uploadFile(uploader, localEquipmentDimTable, "dimension_table_equipment", csvPrefixGlobal)
	uploadFile(uploader, localGameTypeDimTable, "dimension_table_game_types", csvPrefixGlobal)
	uploadFile(uploader, localHitGroupDimTable, "dimension_table_hit_groups", csvPrefixGlobal)

	// save the id state
	saveOutputStateCSV(&startIDState)
	uploadFile(uploader, outputStateCSVName, "global_id_state", csvPrefixBase)

	if !*reprocessFlag && !*subsetReprocessFlag {
		for _, fileName := range filesToMove {
			svc.CopyObject(&s3.CopyObjectInput{
				CopySource: aws.String(bucketName + "/" + fileName),
				Bucket: aws.String(bucketName),
				Key: aws.String(processedPrefix + "/" + filepath.Base(fileName)),
			})
			svc.DeleteObject(&s3.DeleteObjectInput{
				Bucket: aws.String(bucketName),
				Key: aws.String(fileName),
			})
		}
	}
}
