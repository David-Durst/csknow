scp steam@54.219.195.192:csgo-ds/csgo/$1 $2
python disable.py manual
python upload.py manual $2
python list.py manual
cd ../demo_parser
go run cmd/main.go -m
cd ../analytics
./scripts/create_manual_datasets.sh
