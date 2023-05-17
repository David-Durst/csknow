package main

import (
	"flag"
	"fmt"
	d "github.com/David-Durst/csknow/demo_parser/internal"
	c "github.com/David-Durst/csknow/demo_parser/internal/constants"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
	"os"
	"strings"
)

var localDemName string

// prefixes for paths to demos on AWS S3
var demoUnprocessedPrefix = "demos/unprocessed2/"
var demoProcessedPrefix = "demos/processed2/"
var demoProcessedSmallPrefix = "demos/processed2_small/"

// paths to CSVs in AWS
var csvPrefixBase = "demos/csvs3/"
var csvPrefixLocal string
var csvPrefixGlobal string

// these will be used to replace the AWS S3 prefixes if using bot train data set
const trainDemoUnprocessedPrefix = "demos/train_data/unprocessed/"
const trainDemoProcessedPrefix = "demos/train_data/processed/"
const trainCSVPrefixBase = "demos/train_data/csvs/"

// these will be used to replace the AWS S3 prefixes if using big bot train data set
const bigTrainDemoUnprocessedPrefix = "demos/big_train_data/demos/"
const bigTrainDemoProcessedPrefix = "demos/big_train_data/processed/"
const bigTrainCSVPrefixBase = "demos/big_train_data/csvs/"

// these will be used to replace the AWS S3 prefixes if using retakes data set
const retakesDemoUnprocessedPrefix = "demos/retakes_data/unprocessed/"
const retakesDemoProcessedPrefix = "demos/retakes_data/processed/"
const retakesCSVPrefixBase = "demos/retakes_data/csvs/"

// these will be used to replace the AWS S3 prefixes if using bot retakes data set
const botRetakesDemoUnprocessedPrefix = "demos/bot_retakes_data/unprocessed/"
const botRetakesDemoProcessedPrefix = "demos/bot_retakes_data/processed/"
const botRetakesCSVPrefixBase = "demos/bot_retakes_data/csvs/"

// these will be used to replace the AWS S3 prefixes if using manual data set
const manualDemoUnprocessedPrefix = "demos/manual_data/unprocessed/"
const manualDemoProcessedPrefix = "demos/manual_data/processed/"
const manualCSVPrefixBase = "demos/manual_data/csvs/"

const rolloutDemoUnprocessedPrefix = "demos/rollout_data/unprocessed/"
const rolloutDemoProcessedPrefix = "demos/rollout_data/processed/"
const rolloutCSVPrefixBase = "demos/rollout_data/csvs/"

func updatePrefixs() {
	csvPrefixLocal = csvPrefixBase + "local/"
	csvPrefixGlobal = csvPrefixBase + "global/"
}

func main() {
	startIDState := d.DefaultIDState()

	trainDataFlag := flag.Bool("t", false, "set if using bot training data")
	bigTrainDataFlag := flag.Bool("bt", false, "set if using big train data")
	retakesDataFlag := flag.Bool("rd", false, "set if using retakes data")
	botRetakesDataFlag := flag.Bool("brd", false, "set if using retakes data")
	manualDataFlag := flag.Bool("m", false, "set if using manual data")
	rolloutDataFlag := flag.Bool("ro", false, "set if using rollout data")
	// if running locally, skip the aws stuff and just return
	localFlag := flag.Bool("l", false, "set for non-aws (aka local) runs")
	localDemName := flag.String("n", "local.dem", "set for demo's name on local file system")
	flag.Parse()

	shouldFilterRounds := !*manualDataFlag && !*botRetakesDataFlag && !*rolloutDataFlag
	firstRun := true
	if *localFlag {
		d.ParseDemo(*localDemName, *localDemName, &startIDState, firstRun, c.Pro, shouldFilterRounds)
		d.SaveOutputStateCSV(&startIDState)
		os.Exit(0)
	}

	if *trainDataFlag {
		demoUnprocessedPrefix = trainDemoUnprocessedPrefix
		demoProcessedPrefix = trainDemoProcessedPrefix
		csvPrefixBase = trainCSVPrefixBase
		updatePrefixs()
	} else if *bigTrainDataFlag {
		demoUnprocessedPrefix = bigTrainDemoUnprocessedPrefix
		demoProcessedPrefix = bigTrainDemoProcessedPrefix
		csvPrefixBase = bigTrainCSVPrefixBase
		updatePrefixs()
	} else if *manualDataFlag {
		demoUnprocessedPrefix = manualDemoUnprocessedPrefix
		demoProcessedPrefix = manualDemoProcessedPrefix
		csvPrefixBase = manualCSVPrefixBase
		updatePrefixs()
	} else if *retakesDataFlag {
		demoUnprocessedPrefix = retakesDemoUnprocessedPrefix
		demoProcessedPrefix = retakesDemoProcessedPrefix
		csvPrefixBase = retakesCSVPrefixBase
		updatePrefixs()
	} else if *botRetakesDataFlag {
		demoUnprocessedPrefix = botRetakesDemoUnprocessedPrefix
		demoProcessedPrefix = botRetakesDemoProcessedPrefix
		csvPrefixBase = botRetakesCSVPrefixBase
		updatePrefixs()
	} else if *rolloutDataFlag {
		demoUnprocessedPrefix = rolloutDemoUnprocessedPrefix
		demoProcessedPrefix = rolloutDemoProcessedPrefix
		csvPrefixBase = rolloutCSVPrefixBase
		updatePrefixs()
	} else {
		fmt.Printf("please set one of the data set flags\n")
		os.Exit(0)
	}

	sess := session.Must(session.NewSession(&aws.Config{
		Region: aws.String("us-east-1")},
	))

	svc := s3.New(sess)

	downloader := s3manager.NewDownloader(sess)
	uploader := s3manager.NewUploader(sess)

	var filesToMove []string
	demosFolder := demoUnprocessedPrefix

	i := 0
	svc.ListObjectsV2Pages(&s3.ListObjectsV2Input{
		Bucket: aws.String(d.BucketName),
		Prefix: aws.String(demosFolder),
	}, func(p *s3.ListObjectsV2Output, last bool) bool {
		fmt.Printf("Processing page %d\n", i)

		for _, obj := range p.Contents {
			if !strings.HasSuffix(*obj.Key, ".dem") {
				fmt.Printf("Skipping: %s\n", *obj.Key)
				continue
			}
			fmt.Printf("Handling file: %s\n", *obj.Key)
			d.DownloadFile(downloader, *obj.Key, *localDemName)
			d.ParseDemo(*obj.Key, *localDemName, &startIDState, firstRun, c.Pro, shouldFilterRounds)
			firstRun = false
			d.UploadCSVs(uploader, *obj.Key, csvPrefixLocal)
			filesToMove = append(filesToMove, *obj.Key)
		}

		return true
	})

	if len(filesToMove) == 0 {
		os.Exit(0)
	}
}
