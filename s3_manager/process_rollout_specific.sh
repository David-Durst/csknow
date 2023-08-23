scp steam@54.219.195.192:csgo-ds/csgo/$1 $2
python disable.py rollout
python upload.py rollout $2
python list.py rollout
cd ../demo_parser
go run cmd/main.go -ro -dn _$2
cd ../analytics
./scripts/create_rollout_other_datasets.sh $2
