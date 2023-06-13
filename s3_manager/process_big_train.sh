cd ../demo_parser
go run cmd/main.go -bt
cd ../analytics
./scripts/create_big_train_datasets.sh
