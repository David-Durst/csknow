script_dir="tmp"
#https://www.ostricher.com/2014/10/the-right-way-to-get-the-directory-of-a-bash-script/
get_script_dir () {
     SOURCE="${BASH_SOURCE[0]}"
     # While $SOURCE is a symlink, resolve it
     while [ -h "$SOURCE" ]; do
          DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
          SOURCE="$( readlink "$SOURCE" )"
          # If $SOURCE was a relative symlink (so no "/" as prefix, need to resolve it relative to the symlink base directory
          [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
     done
     DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
     script_dir="$DIR"
}
get_script_dir

export pass=$(cat ${script_dir}/../private/.mysql_password)

source $(dirname $(which conda))/../etc/profile.d/conda.sh
conda activate csknow

cd ${script_dir}/csknow-python-analytics/visibility/

rm ${script_dir}/computeVisibilityGenerated.sh
touch ${script_dir}/computeVisibilityGenerated.sh

rm ${script_dir}/recordGameplay.bat
cp ${script_dir}/prefixRecord.bat ${script_dir}/recordGameplay.bat

rm ${script_dir}/recordPlayerState.bat
cp ${script_dir}/prefixRecord.bat ${script_dir}/recordPlayerState.bat

python createHLAEConfigs.py ${script_dir}/../hlae_configs/cfg/9_19_21_walls_entities.txt ${script_dir} ~/third_shared/videos/ ${pass} 1
python createHLAEConfigs.py ${script_dir}/../hlae_configs/cfg/9_20_21_no_wallhacks_entities.txt ${script_dir} ~/third_shared/videos/ ${pass} 0
python createHLAEConfigs.py ${script_dir}/../hlae_configs/cfg/9_20_21_wallhacks_2_entities.txt ${script_dir} ~/third_shared/videos/ ${pass} 1

declare -a ProDemos=(
    "319_titan-epsilon_de_dust2"
    "334_natus-vincere-team-ldlc_de_inferno"
    #"336_natus-vincere-team-ldlc_de_overpass"
    "320_titan-epsilon_de_cache"
    "335_natus-vincere-team-ldlc_de_dust2")
    #"337_natus-vincere-team-ldlc_de_mirage")
for f in "${ProDemos[@]}"; do
  python createHLAEConfigs.py ${script_dir}/../hlae_configs/cfg/${f}_entities.txt ${script_dir} ~/third_shared/videos/ ${pass} 0
done

echo " " >> ${script_dir}/recordGameplay.bat
echo "cd %script_dir%" >> ${script_dir}/recordGameplay.bat

echo " " >> ${script_dir}/recordPlayerState.bat
echo "cd %script_dir%" >> ${script_dir}/recordPlayerState.bat
