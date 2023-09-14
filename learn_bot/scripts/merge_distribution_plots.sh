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


tmp_to_human_histograms=$(mktemp)
tmp_to_human_trajectories=$(mktemp)
tmp_from_human_histograms=$(mktemp)
tmp_from_human_trajectories=$(mktemp)

image_folder=${script_dir}/../learn_bot/latent/analyze/similarity_plots

convert +append ${image_folder}/rollout_learned_vs_all_human_distribution.png \
    ${image_folder}/rollout_handcrafted_vs_all_human_distribution.png \
    ${image_folder}/rollout_default_vs_all_human_distribution.png \
    "${tmp_to_human_histograms}"

convert +append ${image_folder}/rollout_learned_vs_all_human_distribution_trajectories.png \
    ${image_folder}/rollout_handcrafted_vs_all_human_distribution_trajectories.png \
    ${image_folder}/rollout_default_vs_all_human_distribution_trajectories.png \
    "${tmp_to_human_trajectories}"

convert +append ${image_folder}/rollout_all_human_vs_learned_distribution.png \
    ${image_folder}/rollout_all_human_vs_handcrafted_distribution.png \
    ${image_folder}/rollout_all_human_vs_default_distribution.png \
    "${tmp_from_human_histograms}"

convert +append ${image_folder}/rollout_all_human_vs_learned_distribution_trajectories.png \
    ${image_folder}/rollout_all_human_vs_handcrafted_distribution_trajectories.png \
    ${image_folder}/rollout_all_human_vs_default_distribution_trajectories.png \
    "${tmp_from_human_trajectories}"

convert -append "${tmp_to_human_histograms}" "${tmp_from_human_histograms}" \
    ${image_folder}/many_to_from_human_distribution.png

convert -append "${tmp_to_human_trajectories}" "${tmp_from_human_trajectories}" \
    ${image_folder}/many_to_from_human_distribution_trajectories.png

cat ${image_folder}/rollout_learned_vs_all_human_distribution.txt \
    ${image_folder}/rollout_handcrafted_vs_all_human_distribution.txt \
    ${image_folder}/rollout_default_vs_all_human_distribution.txt \
    ${image_folder}/rollout_all_human_vs_learned_distribution.txt \
    ${image_folder}/rollout_all_human_vs_handcrafted_distribution.txt \
    ${image_folder}/rollout_all_human_vs_default_distribution.txt > \
    ${image_folder}/many_to_from_human_distribution.txt 

rm "${tmp_to_human_histograms}" "${tmp_to_human_trajectories}"
rm "${tmp_from_human_histograms}" "${tmp_from_human_trajectories}"
