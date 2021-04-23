package main

import (
	"fmt"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
	"os"
	"path"
)

func downloadDemo(downloader *s3manager.Downloader, fileKey string) {
	// Create a file to write the S3 Object contents to.
	awsF, err := os.Create(localDemName)
	if err != nil {
		fmt.Errorf("failed to create file %q, %v", localDemName, err)
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

func uploadFile(uploader *s3manager.Uploader, csvPath string, fileKey string, local bool) {
	csvFile, err := os.Open(csvPath)
	if err != nil {
		panic(err)
	}
	defer csvFile.Close()
	csvPrefix := csvPrefixGlobal
	if local {
		csvPrefix = csvPrefixLocal
	}

	_, err = uploader.Upload(&s3manager.UploadInput{
		Bucket: aws.String(bucketName),
		Key:    aws.String(csvPrefix + path.Base(fileKey) + ".csv"),
		Body:   csvFile,
	})
	if err != nil {
		fmt.Errorf("Couldn't upload file" + fileKey)
		os.Exit(1)
	}
}

func uploadCSVs(uploader *s3manager.Uploader, fileKey string) {
	uploadFile(uploader, localRoundsCSVName, fileKey + "_rounds", true)
	uploadFile(uploader, localPlayersCSVName, fileKey + "_players", true)
	uploadFile(uploader, localTicksCSVName, fileKey + "_ticks", true)
	uploadFile(uploader, localPlayerAtTickCSVName, fileKey + "_player_at_tick", true)
	uploadFile(uploader, localSpottedCSVName, fileKey + "_spotted", true)
	uploadFile(uploader, localWeaponFireCSVName, fileKey + "_weapon_fire", true)
	uploadFile(uploader, localHurtCSVName, fileKey + "_hurt", true)
	uploadFile(uploader, localGrenadesCSVName, fileKey + "_grenades", true)
	uploadFile(uploader, localGrenadeTrajectoriesCSVName, fileKey + "_grenade_trajectories", true)
	uploadFile(uploader, localPlayerFlashedCSVName, fileKey + "_flashed", true)
	uploadFile(uploader, localPlantsCSVName, fileKey + "_plants", true)
	uploadFile(uploader, localDefusalsCSVName, fileKey + "_defusals", true)
	uploadFile(uploader, localExplosionsCSVName, fileKey + "_explosions", true)
	uploadFile(uploader, localKillsCSVName, fileKey + "_kills", true)
}

