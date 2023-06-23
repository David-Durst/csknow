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
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

var localDemName string

const trainDataName = "train_data"
const bigTrainDataName = "big_train_data"
const allTrainDataName = "all_train_data"
const retakesDataName = "retakes_data"
const botRetakesDataName = "bot_retakes_data"
const manualDataName = "manual_data"
const rolloutDataName = "rollout_data"

const s3UploadScriptPath = "scripts/s3_cp.sh"

const maxDemosPerHDF5 = 50

func runCmd(cmd *exec.Cmd) error {
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func main() {
	startIDState := d.DefaultIDState()

	trainDataFlag := flag.Bool("t", false, "set if using bot training data")
	bigTrainDataFlag := flag.Bool("bt", false, "set if using big train data")
	allTrainDataFlag := flag.Bool("at", false, "set if using all train data")
	retakesDataFlag := flag.Bool("rd", false, "set if using retakes data")
	botRetakesDataFlag := flag.Bool("brd", false, "set if using retakes data")
	manualDataFlag := flag.Bool("m", false, "set if using manual data")
	rolloutDataFlag := flag.Bool("ro", false, "set if using rollout data")
	uploadFlag := flag.Bool("u", false, "set to true if uploading results to s3")
	// if running locally, skip the aws stuff and just return
	localFlag := flag.Bool("l", false, "set for non-aws (aka local) runs")
	deleteLocalDemFlag := flag.Bool("d", true, "set delete local copies of demos")
	localDemName := flag.String("n", "local.dem", "set for demo's name on local file system")
	flag.Parse()

	shouldFilterRounds := !*manualDataFlag && !*botRetakesDataFlag && !*rolloutDataFlag
	firstRun := true
	if *localFlag {
		*localDemName = filepath.Join(c.DemoDirectory, *localDemName)
		d.ParseDemo(*localDemName, *localDemName, &startIDState, firstRun, c.Pro, shouldFilterRounds)
		d.SaveOutputStateCSV(&startIDState)
		os.Exit(0)
	}

	var dataName string
	if *trainDataFlag {
		dataName = trainDataName
	} else if *bigTrainDataFlag {
		dataName = bigTrainDataName
	} else if *allTrainDataFlag {
		dataName = allTrainDataName
	} else if *manualDataFlag {
		dataName = manualDataName
	} else if *retakesDataFlag {
		dataName = retakesDataName
	} else if *botRetakesDataFlag {
		dataName = botRetakesDataName
	} else if *rolloutDataFlag {
		dataName = rolloutDataName
	} else {
		fmt.Printf("please set one of the data set flags\n")
		os.Exit(0)
	}
	dataS3FolderKey := path.Join(d.DemosS3KeyPrefixSuffix, dataName)
	demosS3FolderKey := path.Join(dataS3FolderKey, d.DemosS3KeyPrefixSuffix)
	//badDemosS3FolderKey := path.Join(dataS3FolderKey, d.BadDemosS3KeyPrefixSuffix)
	hdf5S3FolderKey := path.Join(dataS3FolderKey, d.HDF5KeySuffix)

	clearTmpCSVFolder()

	sess := session.Must(session.NewSession(&aws.Config{
		Region: aws.String("us-east-1")},
	))

	svc := s3.New(sess)

	downloader := s3manager.NewDownloader(sess)
	/*
		uploader := s3manager.NewUploader(sess)
		var demosToDelete []string
	*/

	i := 0
	validDemos := 0
	totalDemos := 0
	localDemosAsCSV := 0
	hdf5Index := 0
	svc.ListObjectsV2Pages(&s3.ListObjectsV2Input{
		Bucket: aws.String(d.BucketName),
		Prefix: aws.String(demosS3FolderKey),
	}, func(p *s3.ListObjectsV2Output, last bool) bool {
		fmt.Printf("Processing page %d\n", i)

		for _, obj := range p.Contents {
			if !strings.HasSuffix(*obj.Key, ".dem") {
				fmt.Printf("Skipping: %s\n", *obj.Key)
				continue
			}
			fmt.Printf("handling S3 demo: %s\n", path.Join(d.BucketName, *obj.Key))
			*localDemName = filepath.Join(c.DemoDirectory, filepath.Base(*obj.Key))
			d.DownloadDemo(downloader, *obj.Key, *localDemName)
			if d.ParseDemo(*obj.Key, *localDemName, &startIDState, firstRun, c.Pro, shouldFilterRounds) {
				// only finish first run if demo is valid
				firstRun = false
				validDemos++
				localDemosAsCSV++
				if localDemosAsCSV >= maxDemosPerHDF5 {
					covertCSVToHDF5(dataName, *uploadFlag, dataS3FolderKey, hdf5S3FolderKey, hdf5Index, false)
					clearTmpCSVFolder()
					hdf5Index++
					localDemosAsCSV = 0
					// reset state on each hdf5 write
					startIDState = d.DefaultIDState()
				}
				fmt.Printf("%d valid demos / %d total demos, %d hdf5s\n", validDemos, totalDemos, hdf5Index)
			}
			totalDemos++
			if *deleteLocalDemFlag {
				rmErr := os.Remove(*localDemName)
				if rmErr != nil {
					log.Fatal(rmErr)
				}
			}
			/*
				// reenable when want to deltete invalid demos
				if !d.ParseDemo(*obj.Key, *localDemName, &startIDState, firstRun, c.Pro, shouldFilterRounds) {
					badKey := path.Join(badDemosS3FolderKey, path.Base(*obj.Key))
					d.UploadFile(uploader, *localDemName, badKey)
					demosToDelete = append(demosToDelete, *obj.Key)
				} else {
					// only finish first run if demo is valid
					firstRun = false
				}
			*/
		}

		i++
		return true
	})

	/*
		for _, demoToDelete := range demosToDelete {
			d.DeleteFile(svc, demoToDelete)
		}
	*/

	covertCSVToHDF5(dataName, *uploadFlag, dataS3FolderKey, hdf5S3FolderKey, hdf5Index, true)
	hdf5Index++
	fmt.Printf("%d valid demos / %d total demos, %d hdf5s\n", validDemos, totalDemos, hdf5Index)
}

func clearTmpCSVFolder() {
	if _, err := os.Stat(c.TmpDir); err == nil {
		rmErr := os.RemoveAll(c.TmpDir)
		if rmErr != nil {
			log.Fatal(rmErr)
		}
	}
	_, cpErr := exec.Command("cp", "-r", c.TemplateTmpDir, c.TmpDir).Output()
	if cpErr != nil {
		log.Fatal(cpErr)
	}
}

func covertCSVToHDF5(dataName string, uploadFlag bool, dataS3FolderKey string, hdf5S3FolderKey string, hdf5Index int,
	finalWrite bool) {
	currentPath, err := os.Getwd()
	if err != nil {
		log.Fatal(err)
	}
	csvsPathForConverter := filepath.Join(currentPath, c.TmpDir)
	hdf5FileName := dataName
	if !finalWrite || hdf5Index > 0 {
		hdf5FolderName := filepath.Join(currentPath, c.HDF5Directory, hdf5FileName)
		err = os.MkdirAll(hdf5FolderName, 0777)
		if err != nil {
			log.Fatal(err)
		}
		hdf5FileName = filepath.Join(hdf5FileName, strconv.Itoa(hdf5Index))
	}
	hdf5FileName += ".hdf5"
	hdf5PathForConverter := filepath.Join(currentPath, c.HDF5Directory, hdf5FileName)
	csvToHDF5Path := filepath.Join(currentPath, "..", "analytics", "scripts", "csv_to_hdf5.sh")
	fmt.Println("executing " + csvToHDF5Path + " " + csvsPathForConverter + " " + hdf5PathForConverter)
	csvToHDF5Cmd := exec.Command("/bin/bash", csvToHDF5Path, csvsPathForConverter, hdf5PathForConverter)
	err = runCmd(csvToHDF5Cmd)
	if err != nil {
		log.Fatal(err)
	}

	if uploadFlag {
		hdf5S3CurrentKey := path.Join(dataS3FolderKey, hdf5FileName)
		t := time.Now()
		hdf5S3TemporalKey := path.Join(hdf5S3FolderKey, t.Format("2006_01_02_15_04_05_")+hdf5FileName)
		hdf5UploadCmd := exec.Command("/bin/bash", s3UploadScriptPath, hdf5PathForConverter,
			"s3://csknow/"+hdf5S3CurrentKey)
		err = runCmd(hdf5UploadCmd)
		if err != nil {
			log.Fatal(err)
		}
		hdf5CopyCmd := exec.Command("/bin/bash", s3UploadScriptPath, "s3://csknow/"+hdf5S3CurrentKey,
			"s3://csknow/"+hdf5S3TemporalKey)
		err = runCmd(hdf5CopyCmd)
		if err != nil {
			log.Fatal(err)
		}
		/*
			d.UploadFile(uploader, hdf5PathForConverter, hdf5S3CurrentKey)
			svc.CopyObject(&s3.CopyObjectInput{
				CopySource: aws.String(path.Join(d.BucketName, hdf5S3CurrentKey)),
				Bucket:     aws.String(d.BucketName),
				Key:        aws.String(hdf5S3TemporalKey),
			})
		*/
	}
}
