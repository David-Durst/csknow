package main

import (
	"fmt"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
	"os"
	"path"
)

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

func uploadFile(uploader *s3manager.Uploader, csvPath string, fileKey string, csvPrefix string) {
	csvFile, err := os.Open(csvPath)
	if err != nil {
		panic(err)
	}
	defer csvFile.Close()

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
	uploadFile(uploader, localRoundsCSVName, fileKey + "_rounds", csvPrefixLocal)
	uploadFile(uploader, localPlayersCSVName, fileKey + "_players", csvPrefixLocal)
	uploadFile(uploader, localTicksCSVName, fileKey + "_ticks", csvPrefixLocal)
	uploadFile(uploader, localPlayerAtTickCSVName, fileKey + "_player_at_tick", csvPrefixLocal)
	uploadFile(uploader, localSpottedCSVName, fileKey + "_spotted", csvPrefixLocal)
	uploadFile(uploader, localFootstepCSVName, fileKey + "_footstep", csvPrefixLocal)
	uploadFile(uploader, localWeaponFireCSVName, fileKey + "_weapon_fire", csvPrefixLocal)
	uploadFile(uploader, localHurtCSVName, fileKey + "_hurt", csvPrefixLocal)
	uploadFile(uploader, localGrenadesCSVName, fileKey + "_grenades", csvPrefixLocal)
	uploadFile(uploader, localGrenadeTrajectoriesCSVName, fileKey + "_grenade_trajectories", csvPrefixLocal)
	uploadFile(uploader, localPlayerFlashedCSVName, fileKey + "_flashed", csvPrefixLocal)
	uploadFile(uploader, localPlantsCSVName, fileKey + "_plants", csvPrefixLocal)
	uploadFile(uploader, localDefusalsCSVName, fileKey + "_defusals", csvPrefixLocal)
	uploadFile(uploader, localExplosionsCSVName, fileKey + "_explosions", csvPrefixLocal)
	uploadFile(uploader, localKillsCSVName, fileKey + "_kills", csvPrefixLocal)
}

