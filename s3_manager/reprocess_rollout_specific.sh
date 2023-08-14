python disable.py rollout
python enable.py rollout $1
python list.py rollout
cd ../demo_parser
go run cmd/main.go -ro -dn _$2
cd ../analytics
./scripts/create_rollout_other_datasets.sh $2
