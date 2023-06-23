cd ../demo_parser
go run cmd/main.go -at
cd ../analytics
./scripts/create_all_train_datasets.sh
