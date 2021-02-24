package main

import (
	"bufio"
	"flag"
	"fmt"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/s3/s3manager"
	"log"
	"os"
	"path"
)

const alreadyDownloadedFileName = "already_downloaded.txt"
const processedPrefix = "demos/processed/"
const csvPrefiix = "demos/csvs2/"
const bucketName = "csknow"

func fillAlreadyDownloaded(alreadyDownloaded *map[string]struct{}) {
	alreadyDownloadedFile, err := os.OpenFile(alreadyDownloadedFileName, os.O_APPEND|os.O_CREATE|os.O_RDONLY, 0644)
	if err != nil {
		panic(err)
	}
	defer alreadyDownloadedFile.Close()

	scanner := bufio.NewScanner(alreadyDownloadedFile)
	var exists struct{}
	for scanner.Scan() {
		(*alreadyDownloaded)[scanner.Text()] = exists
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
}

func saveNewlyDownloaded(needToDownload []string) {
	alreadyDownloadedFile, err := os.OpenFile(alreadyDownloadedFileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		panic(err)
	}
	defer alreadyDownloadedFile.Close()

	for _, s := range needToDownload {
		alreadyDownloadedFile.WriteString(s + "\n")
	}
}

func downloadCSVForDemo(downloader *s3manager.Downloader, demKey string, csvType string) {
	// Create a file to write the S3 Object contents to.
	localPath := path.Join("..", "local_data", csvType, demKey + "_" + csvType + ".csv")
	awsF, err := os.Create(localPath)
	if err != nil {
		fmt.Errorf("failed to create file %q, %v", localPath, err)
		os.Exit(1)
	}
	defer awsF.Close()

	// Write the contents of S3 Object to the file
	_, err = downloader.Download(awsF, &s3.GetObjectInput{
		Bucket: aws.String(bucketName),
		Key:    aws.String(csvPrefiix + demKey + "_" + csvType + ".csv"),
	})
	if err != nil {
		fmt.Errorf("failed to download file, %v", err)
		os.Exit(1)
	}

}

func main() {
	alreadyDownloaded := make(map[string]struct{})
	var needToDownload []string
	fillAlreadyDownloaded(&alreadyDownloaded)

	localFlag := flag.Bool("l", false, "set for desktop runs that only download a few csvs")
	flag.Parse()

	sess := session.Must(session.NewSession(&aws.Config{
		Region:      aws.String("us-east-1"),
		Credentials: credentials.AnonymousCredentials,
	}))

	svc := s3.New(sess)

	downloader := s3manager.NewDownloader(sess)

	numDownloaded := 0
	page := 0
	svc.ListObjectsV2Pages(&s3.ListObjectsV2Input{
		Bucket: aws.String(bucketName),
		Prefix: aws.String(processedPrefix + "auto"),
	}, func(p *s3.ListObjectsV2Output, last bool) bool {
		fmt.Printf("Processing page %d\n", page)
		elemsInPage := len(p.Contents)
		curElemInPage := 0
		for _, obj := range p.Contents {
			fmt.Printf("%d / %d in page: %s\n", curElemInPage, elemsInPage, *obj.Key)
			curElemInPage++
			localKey := path.Base(*obj.Key)
			if _, ok := alreadyDownloaded[localKey]; !ok {
				needToDownload = append(needToDownload, localKey)
			}
			downloadCSVForDemo(downloader, localKey,  "position")
			downloadCSVForDemo(downloader, localKey, "spotted")
			downloadCSVForDemo(downloader, localKey, "weapon_fire")
			downloadCSVForDemo(downloader, localKey, "hurt")
			downloadCSVForDemo(downloader, localKey, "grenades")
			downloadCSVForDemo(downloader, localKey, "kills")
			numDownloaded++
			if *localFlag && numDownloaded > 2 {
				return false
			}
		}

		return true
	})

	saveNewlyDownloaded(needToDownload)
}