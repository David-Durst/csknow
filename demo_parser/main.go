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
const localPositionCSVName = "local_position.csv"
const localSpottedCSVName = "local_spotted.csv"
const localWeaponFireCSVName = "local_weapon_fire.csv"
const localHurtCSVName = "local_hurt.csv"
const localGrenadesCSVName = "local_grenades.csv"
const unprocessedPrefix = "demos/unprocessed/"
const processedPrefix = "demos/processed/"
const csvPrefiix = "demos/csvs/"
const bucketName = "csknow"

func main() {

	// if running locally, skip the aws stuff and just return
	localFlag := flag.Bool("l", false, "set for non-aws (aka local) runs")
	flag.Parse()
	if *localFlag {
		processFile("local_run")
		os.Exit(0)
	}


	sess := session.Must(session.NewSession(&aws.Config{
		Region:      aws.String("us-east-1")},
		))

	svc := s3.New(sess)

	downloader := s3manager.NewDownloader(sess)
	uploader := s3manager.NewUploader(sess)

	var filesToMove []string

	i := 0
	svc.ListObjectsV2Pages(&s3.ListObjectsV2Input{
		Bucket: aws.String(bucketName),
		Prefix: aws.String(unprocessedPrefix + "auto"),
	}, func(p *s3.ListObjectsV2Output, last bool) bool {
		fmt.Printf("Processing page %d\n", i)

		for _, obj := range p.Contents {
			fmt.Printf("Handling file: %s\n", *obj.Key)
			downloadFile(downloader, *obj.Key)
			processFile(*obj.Key)
			uploadFile(uploader, *obj.Key)
			filesToMove = append(filesToMove, *obj.Key)
		}

		return true
	})

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
