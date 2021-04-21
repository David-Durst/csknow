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
const localRoundsFile = "local_players.csv"
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
const localDefusalsSVName = "local_defusals.csv"
const localExplosionsCSVName = "local_explosions.csv"
const localKillsCSVName = "local_kills.csv"
const localEquipmentFactTable = "fact_table_equipment.csv"
const localGameTypeFactTable = "fact_table_game_type.csv"
const localHitGroupFactTable = "fact_table_hit_group.csv"
const unprocessedPrefix = "demos/unprocessed/"
const processedPrefix = "demos/processed/"
const csvPrefiix = "demos/csvs2/"
const bucketName = "csknow"

func main() {
	startIDState := IDState{0, 0, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 0, 0, 0}
	firstRun := true

	// if reprocessing, don't move the demos
	reprocessFlag := flag.Bool("r", false, "set for reprocessing demos")
	// if running locally, skip the aws stuff and just return
	localFlag := flag.Bool("l", false, "set for non-aws (aka local) runs")
	flag.Parse()
	gameTypeToID := saveGameTypesFile()
	saveHitGroupsFile()
	if *localFlag {
		processFile("local_run", &startIDState, firstRun, gameTypeToID, "bots")
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
	svc.ListObjectsV2Pages(&s3.ListObjectsV2Input{
		Bucket: aws.String(bucketName),
		Prefix: aws.String(sourcePrefix + "auto"),
	}, func(p *s3.ListObjectsV2Output, last bool) bool {
		fmt.Printf("Processing page %d\n", i)

		for _, obj := range p.Contents {
			fmt.Printf("Handling file: %s\n", *obj.Key)
			downloadDemo(downloader, *obj.Key)
			processFile(*obj.Key, &startIDState, firstRun, gameTypeToID, "bots")
			firstRun = false
			uploadCSVs(uploader, *obj.Key)
			filesToMove = append(filesToMove, *obj.Key)
		}

		return true
	})

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
