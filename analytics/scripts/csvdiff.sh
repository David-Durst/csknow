set -x
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

csvdiffPath="${script_dir}/bin/csvdiff"
if [ ! -f $csvdiffPath ]; then
    ${script_dir}/setup.sh
fi

csvsPath="${script_dir}/../csv_outputs"
olderVersion=$(head -2 "${csvsPath}/versions.txt" | tail -1)
recentVersion=$(head -1 "${csvsPath}/versions.txt")
cat "${csvsPath}/datasets.txt" | while read line
do
    $csvdiffPath "${csvsPath}/${olderVersion}_${line}.csv" "${csvsPath}/${recentVersion}_${line}.csv" -p 0,1
done
