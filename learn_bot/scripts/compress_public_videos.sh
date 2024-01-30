set -x

src_videos_path="/home/durst/Videos/kdenlive/questionnaire/public"
dst_videos_path="/home/durst/Videos/kdenlive/questionnaire/small_public"
human_name=Human
learned_name=CSMoveBot
default_name=CSGOBot
handcrafted_name=RuleBot

mkdir -p $dst_videos_path

compress_videos() {
    src_path=$src_videos_path/$example_name
    dst_path=$dst_videos_path/$example_name
    mkdir -p $dst_path
    ffmpeg -i $src_path/${example_name}_${human_name}.mp4 -hide_banner -loglevel error -y -vcodec libx265 -crf 28 -vf "scale=1280:720" $dst_path/${example_name}_${human_name}.mp4
    ffmpeg -i $src_path/${example_name}_${learned_name}.mp4 -hide_banner -loglevel error -y -vcodec libx265 -crf 28 -vf "scale=1280:720" $dst_path/${example_name}_${learned_name}.mp4
    ffmpeg -i $src_path/${example_name}_${default_name}.mp4 -hide_banner -loglevel error -y -vcodec libx265 -crf 28 -vf "scale=1280:720" $dst_path/${example_name}_${default_name}.mp4
    ffmpeg -i $src_path/${example_name}_${handcrafted_name}.mp4 -hide_banner -loglevel error -y -vcodec libx265 -crf 28 -vf "scale=1280:720" $dst_path/${example_name}_${handcrafted_name}.mp4
}

example_name=example_1
compress_videos

example_name=example_2
compress_videos

example_name=example_3
compress_videos

example_name=example_4
compress_videos

example_name=example_5
compress_videos

example_name=example_6
compress_videos

example_name=example_7
compress_videos

example_name=example_8
compress_videos
