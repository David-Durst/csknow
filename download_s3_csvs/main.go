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
	"os/exec"
	"path"
	"strconv"
	"strings"
)

const gamesCSVName = "global_games.csv"
const localEquipmentDimTable = "dimension_table_equipment.csv"
const localGameTypeDimTable = "dimension_table_game_types.csv"
const localHitGroupDimTable = "dimension_table_hit_groups.csv"
const alreadyDownloadedFileName = "../local_data/already_downloaded.txt"
const processedPrefix = "demos/processed2/"
const processedSmallPrefix = "demos/processed2_small/"
const csvPrefixBase = "demos/csvs3/"
const csvPrefixLocal = csvPrefixBase + "local/"
const csvPrefixGlobal = csvPrefixBase +  "global/"
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

func downloadFile(downloader *s3manager.Downloader, fileKey string, localFileName string) {
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
		fmt.Errorf("failed to download file, %v", err)
		os.Exit(1)
	}
}

func downloadCSVForDemo(downloader *s3manager.Downloader, demKey string, csvType string) {
	localPath := path.Join("..", "local_data", csvType, demKey + "_" + csvType + ".csv")
	downloadFile(downloader, csvPrefixLocal + demKey + "_" + csvType + ".csv", localPath)
}

func main() {
	alreadyDownloaded := make(map[string]struct{})
	var needToDownload []string
	fillAlreadyDownloaded(&alreadyDownloaded)

	localFlag := flag.Bool("l", false, "set for desktop runs that only download a few csvs")
	keyFilterFlag := flag.String("f", "", "set for adding to local runs files that contain a substring")
	subsetFlag := flag.Bool("s", false, "set for server runs that only download a subset csvs (but more than -l)")
	flag.Parse()

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
			downloadCSVForDemo(downloader, localKey, "rounds")
			downloadCSVForDemo(downloader, localKey, "players")
			downloadCSVForDemo(downloader, localKey, "ticks")
			downloadCSVForDemo(downloader, localKey, "player_at_tick")
			downloadCSVForDemo(downloader, localKey, "spotted")
			downloadCSVForDemo(downloader, localKey, "weapon_fire")
			downloadCSVForDemo(downloader, localKey, "hurt")
			downloadCSVForDemo(downloader, localKey, "grenades")
			downloadCSVForDemo(downloader, localKey, "grenade_trajectories")
			downloadCSVForDemo(downloader, localKey, "flashed")
			downloadCSVForDemo(downloader, localKey, "plants")
			downloadCSVForDemo(downloader, localKey, "defusals")
			downloadCSVForDemo(downloader, localKey, "explosions")
			downloadCSVForDemo(downloader, localKey, "kills")
			numDownloaded++
		}

		return true
	})

	localDir := "../local_data/"
	downloadFile(downloader, csvPrefixGlobal + "global_games.csv", localDir + gamesCSVName)
	downloadFile(downloader, csvPrefixGlobal + "dimension_table_equipment.csv", localDir + localEquipmentDimTable)
	downloadFile(downloader, csvPrefixGlobal + "dimension_table_game_types.csv", localDir + localGameTypeDimTable)
	downloadFile(downloader, csvPrefixGlobal + "dimension_table_hit_groups.csv", localDir + localHitGroupDimTable)

	saveNewlyDownloaded(needToDownload)

	fmt.Printf("executing merge.sh")
	out, err := exec.Command("/bin/bash", "merge.sh").Output()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(out)

	if *localFlag {
		fmt.Printf("executing first_lines_games.sh")
		firstLinesGamesOut, err := exec.Command("/bin/bash", "first_lines_games.sh", strconv.Itoa(1 + numDownloaded)).Output()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(firstLinesGamesOut)
	}
}
