package main

import (
	"bufio"
	"flag"
	"fmt"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
	"log"
	"os"
	"path"
	"strings"
)

const gamesCSVName = "global_games.csv"
const localEquipmentDimTable = "dimension_table_equipment.csv"
const localGameTypeDimTable = "dimension_table_game_types.csv"
const localHitGroupDimTable = "dimension_table_hit_groups.csv"

var alreadyDownloadedFileName = "../local_data/already_downloaded.txt"
var dataDir = "local_data"
var dataPath = "../local_data/"

var processedPrefix = "demos/processed2/"
var processedSmallPrefix = "demos/processed2_small/"
var csvPrefixBase = "demos/csvs3/"

// these will be used to replace the prefixes if using bot train data set
const trainProcessedPrefix = "demos/train_data/processed/"
const trainCsvPrefixBase = "demos/train_data/csvs/"

// these will be used to replace the prefixes if using bigTraing data set
const bigTrainProcessedPrefix = "demos/big_train_data/processed/"
const bigTrainCsvPrefixBase = "demos/big_train_data/csvs/"
const bigTrainAlreadyDownloadedFileName = "../big_train_data/already_downloaded.txt"
const bigTrainDir = "big_train_data"
const bigTrainPath = "../big_train_data/"

// these will be used to replace the prefixes if using retakes data set
const retakesProcessedPrefix = "demos/retakes_data/processed/"
const retakesCsvPrefixBase = "demos/retakes_data/csvs/"
const retakesAlreadyDownloadedFileName = "../retakes_data/already_downloaded.txt"
const retakesDir = "retakes_data"
const retakesPath = "../retakes_data/"

// these will be used to replace the prefixes if using bot retakes data set
const botRetakesProcessedPrefix = "demos/bot_retakes_data/processed/"
const botRetakesCsvPrefixBase = "demos/bot_retakes_data/csvs/"
const botRetakesAlreadyDownloadedFileName = "../bot_retakes_data/already_downloaded.txt"
const botRetakesDir = "bot_retakes_data"
const botRetakesPath = "../bot_retakes_data/"

// these will be used to replace the AWS S3 prefixes if using manual data set
const manualProcessedPrefix = "demos/manual_data/processed/"
const manualCsvPrefixBase = "demos/manual_data/csvs/"

var csvPrefixLocal string
var csvPrefixGlobal string

func updatePrefixs() {
	csvPrefixLocal = csvPrefixBase + "local/"
	csvPrefixGlobal = csvPrefixBase + "global/"
}

const bucketName = "csknow"

func fillAlreadyDownloaded(alreadyDownloaded *map[string]struct{}) {
	alreadyDownloadedFile, err := os.OpenFile(alreadyDownloadedFileName, os.O_APPEND|os.O_CREATE|os.O_RDONLY, 0644)
	if err != nil {
		panic(err)
	}
	defer alreadyDownloadedFile.Close()

	scanner := bufio.NewScanner(alreadyDownloadedFile)
	var exists struct{}
	for scanner.Scan() {
		(*alreadyDownloaded)[scanner.Text()] = exists
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
}

func saveNewlyDownloaded(needToDownload []string) {
	alreadyDownloadedFile, err := os.OpenFile(alreadyDownloadedFileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		panic(err)
	}
	defer alreadyDownloadedFile.Close()

	for _, s := range needToDownload {
		alreadyDownloadedFile.WriteString(s + "\n")
	}
}

func downloadFile(downloader *s3manager.Downloader, fileKey string, localFileName string, mustDownload bool) {
	// Create a file to write the S3 Object contents to.
	awsF, err := os.Create(localFileName)
	if err != nil {
		fmt.Errorf("failed to create file %q, %v", localFileName, err)
		os.Exit(1)
	}
	defer awsF.Close()

	// Write the contents of S3 Object to the file
	_, err = downloader.Download(awsF, &s3.GetObjectInput{
		Bucket: aws.String(bucketName),
		Key:    aws.String(fileKey),
	})
	if err != nil {
		if mustDownload {
			fmt.Errorf("failed to download file, %v", err)
			os.Exit(1)
		} else {
			os.Remove(localFileName)
		}
	}
}

func downloadCSVForDemo(downloader *s3manager.Downloader, demKey string, csvType string, mustDownload bool) {
	localPath := path.Join("..", dataDir, csvType, demKey+"_"+csvType+".csv")
	downloadFile(downloader, csvPrefixLocal+demKey+"_"+csvType+".csv", localPath, mustDownload)
}

func main() {
	alreadyDownloaded := make(map[string]struct{})
	var needToDownload []string
	fillAlreadyDownloaded(&alreadyDownloaded)

	trainDataFlag := flag.Bool("t", true, "set -t=false if not using bot training data")
	bigTrainDataFlag := flag.Bool("bt", false, "set if using big train data")
	retakesDataFlag := flag.Bool("rd", false, "set if using retakes data")
	botRetakesDataFlag := flag.Bool("brd", false, "set if using retakes data")
	manualDataFlag := flag.Bool("m", false, "set if using manual data")
	localFlag := flag.Bool("l", false, "set for desktop runs that only download a few csvs")
	keyFilterFlag := flag.String("f", "", "set for adding to local runs files that contain a substring")
	subsetFlag := flag.Bool("s", false, "set for server runs that only download a subset csvs (but more than -l)")
	flag.Parse()

	println("trainDataFlag ", *trainDataFlag)

	if *trainDataFlag {
		processedPrefix = trainProcessedPrefix
		processedSmallPrefix = trainProcessedPrefix
		csvPrefixBase = trainCsvPrefixBase
		updatePrefixs()
	} else if *manualDataFlag {
		processedPrefix = manualProcessedPrefix
		processedSmallPrefix = manualProcessedPrefix
		csvPrefixBase = manualCsvPrefixBase
		updatePrefixs()
	} else if *bigTrainDataFlag {
		processedPrefix = bigTrainProcessedPrefix
		processedSmallPrefix = bigTrainProcessedPrefix
		csvPrefixBase = bigTrainCsvPrefixBase
		alreadyDownloadedFileName = bigTrainAlreadyDownloadedFileName
		dataDir = bigTrainDir
		dataPath = bigTrainPath
	} else if *retakesDataFlag {
		processedPrefix = retakesProcessedPrefix
		processedSmallPrefix = retakesProcessedPrefix
		csvPrefixBase = retakesCsvPrefixBase
		alreadyDownloadedFileName = retakesAlreadyDownloadedFileName
		dataDir = retakesDir
		dataPath = retakesPath
		updatePrefixs()
	} else if *botRetakesDataFlag {
		processedPrefix = botRetakesProcessedPrefix
		processedSmallPrefix = botRetakesProcessedPrefix
		csvPrefixBase = botRetakesCsvPrefixBase
		alreadyDownloadedFileName = botRetakesAlreadyDownloadedFileName
		dataDir = botRetakesDir
		dataPath = botRetakesPath
		updatePrefixs()
	}

	sess := session.Must(session.NewSession(&aws.Config{
		Region:      aws.String("us-east-1"),
		Credentials: credentials.AnonymousCredentials,
	}))

	svc := s3.New(sess)

	downloader := s3manager.NewDownloader(sess)

	awsPrefix := processedPrefix
	if *subsetFlag || *localFlag {
		awsPrefix = processedSmallPrefix
	}
	numDownloaded := 0
	page := 0
	svc.ListObjectsV2Pages(&s3.ListObjectsV2Input{
		Bucket: aws.String(bucketName),
		Prefix: aws.String(awsPrefix),
	}, func(p *s3.ListObjectsV2Output, last bool) bool {
		fmt.Printf("Processing page %d\n", page)
		elemsInPage := len(p.Contents)
		curElemInPage := 0
		for _, obj := range p.Contents {
			if !strings.HasSuffix(*obj.Key, ".dem") {
				fmt.Printf("Skipping %d / %d in page: %s\n", curElemInPage, elemsInPage, *obj.Key)
				curElemInPage++
				continue
			}
			localKey := path.Base(*obj.Key)
			if *localFlag && !(numDownloaded <= 2 || (*keyFilterFlag != "" && strings.Contains(localKey, *keyFilterFlag))) {
				curElemInPage++
				continue
			}
			fmt.Printf("%d / %d in page: %s\n", curElemInPage, elemsInPage, *obj.Key)
			curElemInPage++
			if _, ok := alreadyDownloaded[localKey]; !ok {
				needToDownload = append(needToDownload, localKey)
			}
			downloadCSVForDemo(downloader, localKey, "unfiltered_rounds", true)
			downloadCSVForDemo(downloader, localKey, "filtered_rounds", true)
			downloadCSVForDemo(downloader, localKey, "players", true)
			downloadCSVForDemo(downloader, localKey, "ticks", true)
			downloadCSVForDemo(downloader, localKey, "player_at_tick", true)
			downloadCSVForDemo(downloader, localKey, "spotted", true)
			downloadCSVForDemo(downloader, localKey, "footstep", true)
			downloadCSVForDemo(downloader, localKey, "weapon_fire", true)
			downloadCSVForDemo(downloader, localKey, "hurt", true)
			downloadCSVForDemo(downloader, localKey, "grenades", true)
			downloadCSVForDemo(downloader, localKey, "grenade_trajectories", true)
			downloadCSVForDemo(downloader, localKey, "flashed", true)
			downloadCSVForDemo(downloader, localKey, "plants", true)
			downloadCSVForDemo(downloader, localKey, "defusals", true)
			downloadCSVForDemo(downloader, localKey, "explosions", true)
			downloadCSVForDemo(downloader, localKey, "say", true)
			downloadCSVForDemo(downloader, localKey, "kills", true)
			//downloadCSVForDemo(downloader, localKey, "skill", false)
			numDownloaded++
		}

		return true
	})

	localDir := dataPath
	downloadFile(downloader, csvPrefixGlobal+"global_games.csv", localDir+gamesCSVName, true)
	downloadFile(downloader, csvPrefixGlobal+"dimension_table_equipment.csv", localDir+localEquipmentDimTable, true)
	downloadFile(downloader, csvPrefixGlobal+"dimension_table_game_types.csv", localDir+localGameTypeDimTable, true)
	downloadFile(downloader, csvPrefixGlobal+"dimension_table_hit_groups.csv", localDir+localHitGroupDimTable, true)

	saveNewlyDownloaded(needToDownload)

	/*
		if *retakesDataFlag {
			fmt.Printf("executing merge_retakes.sh")
			out, err := exec.Command("/bin/bash", "merge_retakes.sh").Output()
			if err != nil {
				log.Fatal(err)
			}
			fmt.Println(out)
		} else if *botRetakesDataFlag {
			fmt.Printf("executing merge_bot_retakes.sh")
			out, err := exec.Command("/bin/bash", "merge_bot_retakes.sh").Output()
			if err != nil {
				log.Fatal(err)
			}
			fmt.Println(out)
		} else {
			fmt.Printf("executing merge.sh")
			out, err := exec.Command("/bin/bash", "merge.sh").Output()
			if err != nil {
				log.Fatal(err)
			}
			fmt.Println(out)
		}

		if *localFlag {
			fmt.Printf("executing first_lines_games.sh")
			firstLinesGamesOut, err := exec.Command("/bin/bash", "first_lines_games.sh", strconv.Itoa(1+numDownloaded)).Output()
			if err != nil {
				log.Fatal(err)
			}
			fmt.Println(firstLinesGamesOut)
		}
	*/
}
