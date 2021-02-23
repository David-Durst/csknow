package main

import (
	"fmt"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
	"os"
	"path"
)

func downloadFile(downloader *s3manager.Downloader, fileKey string) {
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

func uploadFile(uploader *s3manager.Uploader, fileKey string) {
	csvFile, err := os.Open(localCSVName)
	if err != nil {
		panic(err)
	}
	defer csvFile.Close()

	_, err = uploader.Upload(&s3manager.UploadInput{
		Bucket: aws.String(bucketName),
		Key:    aws.String(csvPrefiix + path.Base(fileKey) + ".csv"),
		Body:   csvFile,
	})
	if err != nil {
		fmt.Errorf("Couldn't upload file" + fileKey)
		os.Exit(1)
	}
}

