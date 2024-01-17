set -x
cut_video() {
    ffmpeg -i ${videos_path}/${src_video} -ss ${start_time} -to ${end_time} -hide_banner -loglevel error -y -async 1 -strict -2 ${videos_path}/${dst_video}
    #ffmpeg -i ${videos_path}/${src_video} -ss ${start_time} -to ${end_time} -y -async 1 -strict -2 ${videos_path}/${dst_video}
}

#learned videos
videos_path="/home/durst/Videos/kdenlive/questionnaire/learned"
learned_main_video="learned_main.mkv"

# example 1
src_video=$learned_main_video
start_time="00:00:02.367"
end_time="00:00:07.800"
dst_video="example-1-learned.mp4"
cut_video

# example 2
start_time="00:00:17.900"
end_time="00:00:30.500"
dst_video="example-2-learned.mp4"
cut_video

# example 3
start_time="00:00:41.417"
end_time="00:00:59.267"
dst_video="example-3-learned.mp4"
cut_video

# example 4
start_time="00:01:07.500"
end_time="00:01:21.517"
dst_video="example-4-learned.mp4"
cut_video

# example 5
start_time="00:01:29.633"
end_time="00:01:45.467"
dst_video="example-5-learned.mp4"
cut_video

# example 6
start_time="00:01:51.000"
end_time="00:02:31.317"
dst_video="example-6-learned.mp4"
cut_video

# example 7
start_time="00:02:32.300"
end_time="00:03:01.850"
dst_video="example-7-learned.mp4"
cut_video

# example 8
start_time="00:03:02.717"
end_time="00:03:43.117"
dst_video="example-8-learned.mp4"
cut_video

#hand-crafted videos
videos_path="/home/durst/Videos/kdenlive/questionnaire/handcrafted"
handcrafted_main_video="handcrafted_main.mkv"

# example 1
src_video=$handcrafted_main_video
start_time="00:00:04.800"
end_time="00:00:18.600"
dst_video="example-1-handcrafted.mp4"
cut_video

# example 2
start_time="00:00:28.983"
end_time="00:00:37"
dst_video="example-2-handcrafted.mp4"
cut_video

# example 3
start_time="00:00:51.617"
end_time="00:00:59"
dst_video="example-3-handcrafted.mp4"
cut_video

# example 4
start_time="00:01:12.767"
end_time="00:01:17.617"
dst_video="example-4-handcrafted.mp4"
cut_video

# example 5
start_time="00:01:32.267"
end_time="00:01:35.500"
dst_video="example-5-handcrafted.mp4"
cut_video

# example 6
start_time="00:01:36.433"
end_time="00:01:48"
dst_video="example-6-handcrafted.mp4"
cut_video

# example 7
start_time="00:02:03.717"
end_time="00:02:11.117"
dst_video="example-7-handcrafted.mp4"
cut_video

# example 8
start_time="00:02:12.067"
end_time="00:02:52.333"
dst_video="example-8-handcrafted.mp4"
cut_video

#default videos
videos_path="/home/durst/Videos/kdenlive/questionnaire/default"
default_main_video="default_main.mkv"

# example 1
src_video=$default_main_video
start_time="00:00:09.417"
end_time="00:00:30.917"
dst_video="example-1-default.mp4"
cut_video

# example 2
start_time="00:00:31.800"
end_time="00:00:56.167"
dst_video="example-2-default.mp4"
cut_video

# example 3
start_time="00:00:56.817"
end_time="00:01:27.533"
dst_video="example-3-default.mp4"
cut_video

# example 4
start_time="00:01:28.233"
end_time="00:01:47.933"
dst_video="example-4-default.mp4"
cut_video

# example 5
start_time="00:01:57.167"
end_time="00:02:13.383"
dst_video="example-5-default.mp4"
cut_video

# example 6
start_time="00:02:14.150"
end_time="00:02:52.283"
dst_video="example-6-default.mp4"
cut_video

# example 7
start_time="00:02:53.083"
end_time="00:03:10.433"
dst_video="example-7-default.mp4"
cut_video

# example 8
start_time="00:03:11.117"
end_time="00:03:33.767"
dst_video="example-8-default.mp4"
cut_video
