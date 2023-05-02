package internal

import (
	"fmt"
	c "github.com/David-Durst/csknow/demo_parser/internal/constants"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
	"os"
	"path"
	"path/filepath"
)

const BucketName = "csknow"

func DownloadFile(downloader *s3manager.Downloader, fileKey string, localFileName string) {
	// Create a file to write the S3 Object contents to.
	awsF, err := os.Create(localFileName)
	if err != nil {
		fmt.Errorf("failed to create file %q, %v", localFileName, err)
		os.Exit(1)
	}
	defer awsF.Close()

	// Write the contents of S3 Object to the file
	_, err = downloader.Download(awsF, &s3.GetObjectInput{
		Bucket: aws.String(BucketName),
		Key:    aws.String(fileKey),
	})
	if err != nil {
		fmt.Errorf("failed to download file, %v", err)
		os.Exit(1)
	}
}

func UploadFile(uploader *s3manager.Uploader, csvPath string, fileKey string, csvPrefix string) {
	csvFile, err := os.Open(filepath.Join(c.TmpDir, csvPath))
	if err != nil {
		panic(err)
	}
	defer csvFile.Close()

	_, err = uploader.Upload(&s3manager.UploadInput{
		Bucket: aws.String(BucketName),
		Key:    aws.String(csvPrefix + path.Base(fileKey) + ".csv"),
		Body:   csvFile,
	})
	if err != nil {
		fmt.Errorf("Couldn't upload file" + fileKey)
		os.Exit(1)
	}
}

func UploadCSVs(uploader *s3manager.Uploader, fileKey string, csvPrefixLocal string) {
	UploadFile(uploader, c.LocalUnfilteredRoundsCSVName, fileKey+"_unfiltered_rounds", csvPrefixLocal)
	UploadFile(uploader, c.LocalFilteredRoundsCSVName, fileKey+"_filtered_rounds", csvPrefixLocal)
	UploadFile(uploader, c.LocalPlayersCSVName, fileKey+"_players", csvPrefixLocal)
	UploadFile(uploader, c.LocalTicksCSVName, fileKey+"_ticks", csvPrefixLocal)
	UploadFile(uploader, c.LocalPlayerAtTickCSVName, fileKey+"_player_at_tick", csvPrefixLocal)
	UploadFile(uploader, c.LocalSpottedCSVName, fileKey+"_spotted", csvPrefixLocal)
	UploadFile(uploader, c.LocalFootstepCSVName, fileKey+"_footstep", csvPrefixLocal)
	UploadFile(uploader, c.LocalWeaponFireCSVName, fileKey+"_weapon_fire", csvPrefixLocal)
	UploadFile(uploader, c.LocalHurtCSVName, fileKey+"_hurt", csvPrefixLocal)
	UploadFile(uploader, c.LocalGrenadesCSVName, fileKey+"_grenades", csvPrefixLocal)
	UploadFile(uploader, c.LocalGrenadeTrajectoriesCSVName, fileKey+"_grenade_trajectories", csvPrefixLocal)
	UploadFile(uploader, c.LocalPlayerFlashedCSVName, fileKey+"_flashed", csvPrefixLocal)
	UploadFile(uploader, c.LocalPlantsCSVName, fileKey+"_plants", csvPrefixLocal)
	UploadFile(uploader, c.LocalDefusalsCSVName, fileKey+"_defusals", csvPrefixLocal)
	UploadFile(uploader, c.LocalExplosionsCSVName, fileKey+"_explosions", csvPrefixLocal)
	UploadFile(uploader, c.LocalSayCSVName, fileKey+"_say", csvPrefixLocal)
	UploadFile(uploader, c.LocalKillsCSVName, fileKey+"_kills", csvPrefixLocal)
}
