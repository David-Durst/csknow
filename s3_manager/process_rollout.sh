conda activate s3_manager
scp steam@54.219.195.192:csgo-ds/csgo/$1 $2
python disable.py rollout
python upload.py rollout $2
python list.py rollout
cd ../demo_parser
go run cmd/main.go -ro
conda deactivate
