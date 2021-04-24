package main

import (
	"flag"
	"fmt"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
	"os"
	"path/filepath"
)

const localDemName = "local.dem"
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
const csvPrefixLocal = "demos/csvs3/local/"
const csvPrefixGlobal = "demos/csvs3/global/"
var gameTypes = [2]string{"pros","bots"}
const bucketName = "csknow"

func main() {
	startIDState := IDState{0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 0}
	firstRun := true

	// if reprocessing, don't move the demos
	reprocessFlag := flag.Bool("r", false, "set for reprocessing demos")
	// if running locally, skip the aws stuff and just return
	localFlag := flag.Bool("l", false, "set for non-aws (aka local) runs")
	flag.Parse()
	saveGameTypesFile()
	saveHitGroupsFile()
	if *localFlag {
		processFile("local_run", &startIDState, firstRun, 1)
		firstRun = false
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
	}

	i := 0
	for gameTypeIndex, gameTypeString := range gameTypes {
		sourcePrefixWithType := sourcePrefix + gameTypeString + "/"
		svc.ListObjectsV2Pages(&s3.ListObjectsV2Input{
			Bucket: aws.String(bucketName),
			Prefix: aws.String(sourcePrefixWithType),
		}, func(p *s3.ListObjectsV2Output, last bool) bool {
			fmt.Printf("Processing page %d\n", i)

			for _, obj := range p.Contents {
				fmt.Printf("Handling file: %s\n", *obj.Key)
				downloadDemo(downloader, *obj.Key)
				processFile(*obj.Key, &startIDState, firstRun, gameTypeIndex)
				firstRun = false
				uploadCSVs(uploader, *obj.Key)
				filesToMove = append(filesToMove, *obj.Key)
			}

			return true
		})
	}
	uploadFile(uploader, gamesCSVName, "global_games", false)
	uploadFile(uploader, localEquipmentDimTable, "dimension_table_equipment", false)
	uploadFile(uploader, localGameTypeDimTable, "dimension_table_game_types", false)
	uploadFile(uploader, localHitGroupDimTable, "dimension_table_hit_groups", false)

	if !*reprocessFlag {
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
