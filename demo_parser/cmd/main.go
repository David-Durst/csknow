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
	"path/filepath"
	"strings"
)

var localDemName string
var unprocessedPrefix = "demos/unprocessed2/"
var processedPrefix = "demos/processed2/"
var processedSmallPrefix = "demos/processed2_small/"
var csvPrefixBase = "demos/csvs3/"

// these will be used to replace the prefixes if using bot train data set
const trainUnprocessedPrefix = "demos/train_data/unprocessed/"
const trainProcessedPrefix = "demos/train_data/processed/"
const trainCsvPrefixBase = "demos/train_data/csvs/"

// paths to CSVs in AWS
var csvPrefixLocal string
var csvPrefixGlobal string

func updatePrefixs() {
	csvPrefixLocal = csvPrefixBase + "local/"
	csvPrefixGlobal = csvPrefixBase + "global/"
}

func main() {
	startIDState := d.DefaultIDState()

	trainDataFlag := flag.Bool("t", true, "set if not using bot training data")
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

	d.SaveGameTypesFile()
	d.SaveHitGroupsFile()
	if *localFlag {
		if !firstRun {
			startIDState = d.ParseInputStateCSV()
		}
		d.ProcessStructure(*localDemName, *localDemName, &startIDState, firstRun, 1)
		d.FilterRounds(&startIDState)
		d.SaveStructure(&startIDState, firstRun)
		d.SaveOutputStateCSV(&startIDState)
		os.Exit(0)
	}

	if *trainDataFlag {
		unprocessedPrefix = trainUnprocessedPrefix
		processedPrefix = trainProcessedPrefix
		csvPrefixBase = trainCsvPrefixBase
		updatePrefixs()
	}

	sess := session.Must(session.NewSession(&aws.Config{
		Region: aws.String("us-east-1")},
	))

	svc := s3.New(sess)

	downloader := s3manager.NewDownloader(sess)
	uploader := s3manager.NewUploader(sess)

	var filesToMove []string
	var destinationAppendix []string
	sourcePrefix := unprocessedPrefix
	if *reprocessFlag {
		sourcePrefix = processedPrefix
	} else if *subsetReprocessFlag {
		sourcePrefix = processedSmallPrefix
	}

	idStateAWS := csvPrefixBase + c.BaseStateCSVName
	result, err := svc.ListObjectsV2(&s3.ListObjectsV2Input{
		Bucket: aws.String(d.BucketName),
		Prefix: &idStateAWS,
	})
	if err != nil {
		panic(err)
	}

	// if not reprocessing and already have an id state, start from there
	if *result.KeyCount == 1 && !*reprocessFlag && !*subsetReprocessFlag {
		d.DownloadFile(downloader, *result.Contents[0].Key, c.InputStateCSVName)
		startIDState = d.ParseInputStateCSV()
		// set first run to false since found an old state and not reprocessing
		firstRun = false
	}

	gamesAWS := csvPrefixGlobal + c.GlobalGamesCSVName
	gamesResult, gamesErr := svc.ListObjectsV2(&s3.ListObjectsV2Input{
		Bucket: aws.String(d.BucketName),
		Prefix: &gamesAWS,
	})
	if gamesErr != nil {
		panic(gamesErr)
	}

	// if not reprocessing and already have an games file, start from there
	if *gamesResult.KeyCount == 1 && !*reprocessFlag && !*subsetReprocessFlag {
		d.DownloadFile(downloader, *gamesResult.Contents[0].Key, c.GlobalGamesCSVName)
	}

	i := 0
	localGameTypes := c.GameTypes()
	for gameTypeIndex, gameTypeString := range localGameTypes {
		sourcePrefixWithType := sourcePrefix + gameTypeString + "/"
		svc.ListObjectsV2Pages(&s3.ListObjectsV2Input{
			Bucket: aws.String(d.BucketName),
			Prefix: aws.String(sourcePrefixWithType),
		}, func(p *s3.ListObjectsV2Output, last bool) bool {
			fmt.Printf("Processing page %d\n", i)

			for _, obj := range p.Contents {
				if !strings.HasSuffix(*obj.Key, ".dem") {
					fmt.Printf("Skipping: %s\n", *obj.Key)
					continue
				}
				fmt.Printf("Handling file: %s\n", *obj.Key)
				d.DownloadFile(downloader, *obj.Key, *localDemName)
				d.ProcessStructure(*obj.Key, *localDemName, &startIDState, firstRun, c.GameType(gameTypeIndex))
				d.FilterRounds(&startIDState)
				d.SaveStructure(&startIDState, firstRun)
				firstRun = false
				d.UploadCSVs(uploader, *obj.Key, csvPrefixLocal)
				filesToMove = append(filesToMove, *obj.Key)
				destinationAppendix = append(destinationAppendix, gameTypeString+"/")
			}

			return true
		})
	}

	if len(filesToMove) == 0 {
		os.Exit(0)
	}

	d.UploadFile(uploader, c.GlobalGamesCSVName, "global_games", csvPrefixGlobal)
	d.UploadFile(uploader, c.LocalEquipmentDimTable, "dimension_table_equipment", csvPrefixGlobal)
	d.UploadFile(uploader, c.LocalGameTypeDimTable, "dimension_table_game_types", csvPrefixGlobal)
	d.UploadFile(uploader, c.LocalHitGroupDimTable, "dimension_table_hit_groups", csvPrefixGlobal)

	// save the id state
	d.SaveOutputStateCSV(&startIDState)
	d.UploadFile(uploader, c.OutputStateCSVName, "global_id_state", csvPrefixBase)

	if !*reprocessFlag && !*subsetReprocessFlag {
		for i, fileName := range filesToMove {
			svc.CopyObject(&s3.CopyObjectInput{
				CopySource: aws.String(d.BucketName + "/" + fileName),
				Bucket:     aws.String(d.BucketName),
				Key:        aws.String(processedPrefix + destinationAppendix[i] + filepath.Base(fileName)),
			})
			svc.DeleteObject(&s3.DeleteObjectInput{
				Bucket: aws.String(d.BucketName),
				Key:    aws.String(fileName),
			})
		}
	}
}
