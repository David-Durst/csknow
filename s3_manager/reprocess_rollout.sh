python list.py rollout
cd ../demo_parser
go run cmd/main.go -ro
cd ../analytics
./scripts/create_rollout_datasets.sh
