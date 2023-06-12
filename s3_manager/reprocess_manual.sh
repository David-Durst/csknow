python disable.py manual
python enable.py manual $1
python list.py manual
cd ../demo_parser
go run cmd/main.go -m
cd ../analytics
./scripts/create_manual_datasets.sh
