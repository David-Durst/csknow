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
	"strings"
	"time"
)

var localDemName string

const trainDataName = "train_data"
const bigTrainDataName = "big_train_data"
const retakesDataName = "retakes_data"
const botRetakesDataName = "bot_retakes_data"
const manualDataName = "manual_data"
const rolloutDataName = "rollout_data"

func main() {
	startIDState := d.DefaultIDState()

	trainDataFlag := flag.Bool("t", false, "set if using bot training data")
	bigTrainDataFlag := flag.Bool("bt", false, "set if using big train data")
	retakesDataFlag := flag.Bool("rd", false, "set if using retakes data")
	botRetakesDataFlag := flag.Bool("brd", false, "set if using retakes data")
	manualDataFlag := flag.Bool("m", false, "set if using manual data")
	rolloutDataFlag := flag.Bool("ro", false, "set if using rollout data")
	uploadFlag := flag.Bool("u", true, "set to false if not uploading results to s3")
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

	var dataName string
	if *trainDataFlag {
		dataName = trainDataName
	} else if *bigTrainDataFlag {
		dataName = bigTrainDataName
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
	hdf5S3FolderKey := path.Join(dataS3FolderKey, d.HDF5KeySuffix)

	// get local data folder ready
	if _, err := os.Stat(c.TmpDir); err == nil {
		rmErr := os.RemoveAll(c.TmpDir)
		if rmErr != nil {
			log.Println(rmErr)
			return
		}
	}
	_, cpErr := exec.Command("cp", "-r", c.TemplateTmpDir, c.TmpDir).Output()
	if cpErr != nil {
		log.Fatal(cpErr)
	}

	sess := session.Must(session.NewSession(&aws.Config{
		Region: aws.String("us-east-1")},
	))

	svc := s3.New(sess)

	downloader := s3manager.NewDownloader(sess)

	i := 0
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
			d.ParseDemo(*obj.Key, *localDemName, &startIDState, firstRun, c.Pro, shouldFilterRounds)
			firstRun = false
		}

		i++
		return true
	})

	currentPath, err := os.Getwd()
	if err != nil {
		log.Println(err)
	}
	csvsPathForConverter := filepath.Join(currentPath, c.TmpDir)
	hdf5FileName := dataName + ".hdf5"
	hdf5PathForConverter := filepath.Join(currentPath, c.HDF5Directory, hdf5FileName)
	csvToHDF5Path := filepath.Join(currentPath, "..", "analytics", "scripts", "csv_to_hdf5.sh")
	fmt.Println("executing " + csvToHDF5Path + " " + csvsPathForConverter + " " + hdf5PathForConverter)
	out, err := exec.Command("/bin/bash", csvToHDF5Path, csvsPathForConverter, hdf5PathForConverter).Output()
	if err != nil {
		log.Println(string(out))
		log.Fatal(err)
	}
	fmt.Println(string(out))

	if *uploadFlag {
		uploader := s3manager.NewUploader(sess)
		hdf5S3CurrentKey := path.Join(dataS3FolderKey, hdf5FileName)
		t := time.Now()
		hdf5S3TemporalKey := path.Join(hdf5S3FolderKey, t.Format("2006_01_02_15_04_05_")+hdf5FileName)
		d.UploadFile(uploader, hdf5PathForConverter, hdf5S3CurrentKey)
		svc.CopyObject(&s3.CopyObjectInput{
			CopySource: aws.String(path.Join(d.BucketName, hdf5S3CurrentKey)),
			Bucket:     aws.String(d.BucketName),
			Key:        aws.String(hdf5S3TemporalKey),
		})
	}
}
