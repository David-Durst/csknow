package internal

import (
	"fmt"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
	"os"
)

const BucketName = "csknow"
const DemosS3KeyPrefixSuffix = "demos"
const HDF5KeySuffix = "hdf5"

func DownloadDemo(downloader *s3manager.Downloader, fileKey string, localFileName string) {
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

func UploadFile(uploader *s3manager.Uploader, localPath string, s3Key string) {
	csvFile, err := os.Open(localPath)
	if err != nil {
		panic(err)
	}
	defer csvFile.Close()

	_, err = uploader.Upload(&s3manager.UploadInput{
		Bucket: aws.String(BucketName),
		Key:    aws.String(s3Key),
		Body:   csvFile,
	})
	if err != nil {
		fmt.Errorf("Couldn't upload file" + localPath)
		os.Exit(1)
	}
}
