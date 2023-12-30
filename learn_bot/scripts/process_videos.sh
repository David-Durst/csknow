set -x
cut_video() {
    #ffmpeg -i ${videos_path}/${src_video} -ss ${start_time} -to ${end_time} -hide_banner -loglevel error -y -async 1 -strict -2 ${videos_path}/${dst_video}
    ffmpeg -i ${videos_path}/${src_video} -ss ${start_time} -to ${end_time} -y -async 1 -strict -2 ${videos_path}/${dst_video}
}

#learned videos
videos_path="/home/durst/Videos/kdenlive/questionnaire/learned"
learned_main_video="learned_main.mkv"
learned_5_fix_video="learned_5_fix.mkv"
learned_7_fix_video="learned_7_fix.mkv"

# example 1
src_video=$learned_main_video
start_time="00:00:10.817"
end_time="00:00:22"
dst_video="example-1-learned.mp4"
cut_video
exit 0
# example 2
start_time="00:00:32.883"
end_time="00:00:50"
dst_video="example-2-learned.mp4"
cut_video

# example 3
start_time="00:00:57.667"
end_time="00:01:14.767"
dst_video="example-3-learned.mp4"
cut_video

# example 4
start_time="00:01:24.933"
end_time="00:02:05.333"
dst_video="example-4-learned.mp4"
cut_video

# example 5
src_video=$learned_5_fix_video
start_time="00:01:27.050"
end_time="00:01:42.050"
dst_video="example-5-learned.mp4"
cut_video

# example 6
src_video=$learned_main_video
start_time="00:02:47.583"
end_time="00:02:58.983"
dst_video="example-6-learned.mp4"
cut_video

# example 7
src_video=$learned_7_fix_video
start_time="00:00:38.933"
end_time="00:01:20.217"
dst_video="example-7-learned.mp4"
cut_video

# example 8
src_video=$learned_main_video
start_time="00:03:48.233"
end_time="00:03:52.500"
dst_video="example-8-learned.mp4"
cut_video

#hand-crafted videos
videos_path="/home/durst/Videos/kdenlive/questionnaire/handcrafted"
handcrafted_main_video="handcrafted_main.mkv"

# example 1
src_video=$handcrafted_main_video
start_time="00:00:04.483"
end_time="00:00:10.167"
dst_video="example-1-handcrafted.mp4"
cut_video

# example 2
start_time="00:00:28.150"
end_time="00:00:33"
dst_video="example-2-handcrafted.mp4"
cut_video

# example 3
start_time="00:00:46.567"
end_time="00:00:52.350"
dst_video="example-3-handcrafted.mp4"
cut_video

# example 4
start_time="00:00:53.317"
end_time="00:01:33.700"
dst_video="example-4-handcrafted.mp4"
cut_video

# example 5
start_time="00:01:34.517"
end_time="00:01:38"
dst_video="example-5-handcrafted.mp4"
cut_video

# example 6
start_time="00:01:49.700"
end_time="00:01:53.683"
dst_video="example-6-handcrafted.mp4"
cut_video

# example 7
start_time="00:01:54.517"
end_time="00:02:04.133"
dst_video="example-7-handcrafted.mp4"
cut_video

# example 8
start_time="00:02:05"
end_time="00:02:06.550"
dst_video="example-8-handcrafted.mp4"
cut_video

#default videos
videos_path="/home/durst/Videos/kdenlive/questionnaire/default"
default_main_video="default_main.mkv"

# example 1
src_video=$default_main_video
start_time="00:00:03.550"
end_time="00:00:22"
dst_video="example-1-default.mp4"
cut_video

# example 2
start_time="00:00:28.400"
end_time="00:00:45"
dst_video="example-2-default.mp4"
cut_video

# example 3
start_time="00:00:54.283"
end_time="00:01:19"
dst_video="example-3-default.mp4"
cut_video

# example 4
start_time="00:01:32.517"
end_time="00:01:57"
dst_video="example-4-default.mp4"
cut_video

# example 5
start_time="00:01:58.367"
end_time="00:02:19.583"
dst_video="example-5-default.mp4"
cut_video

# example 6
start_time="00:02:28.233"
end_time="00:02:40"
dst_video="example-6-default.mp4"
cut_video

# example 7
start_time="00:02:47.150"
end_time="00:03:26.233"
dst_video="example-7-default.mp4"
cut_video

# example 8
start_time="00:03:26.950"
end_time="00:03:33.450"
dst_video="example-8-default.mp4"
cut_video
