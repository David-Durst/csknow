src_videos_path="/home/durst/Videos/kdenlive/questionnaire"
dst_videos_path="/home/durst/Videos/kdenlive/questionnaire/public_renamed"
human_name=Human
learned_name=MLMove
default_name=GameBot
handcrafted_name=RuleMove

mkdir -p $dst_videos_path

#['human', 'default', 'learned', 'hand-crafted']
example_name=example_1
example_path=$dst_videos_path/$example_name
mkdir -p $example_path
cp $src_videos_path/human/example-1-human-g2-vs-nip.mp4 $example_path/${example_name}_${human_name}.mp4
cp $src_videos_path/learned/example-1-learned.mp4 $example_path/${example_name}_${learned_name}.mp4
cp $src_videos_path/default/example-1-default.mp4 $example_path/${example_name}_${default_name}.mp4
cp $src_videos_path/handcrafted/example-1-handcrafted.mp4 $example_path/${example_name}_${handcrafted_name}.mp4

#['hand-crafted', 'human', 'learned', 'default']
example_name=example_2
example_path=$dst_videos_path/$example_name
mkdir -p $example_path
cp $src_videos_path/human/example-2-human-chococheck-vs-mythic.mp4 $example_path/${example_name}_${human_name}.mp4
cp $src_videos_path/learned/example-2-learned.mp4 $example_path/${example_name}_${learned_name}.mp4
cp $src_videos_path/default/example-2-default.mp4 $example_path/${example_name}_${default_name}.mp4
cp $src_videos_path/handcrafted/example-2-handcrafted.mp4 $example_path/${example_name}_${handcrafted_name}.mp4

#['learned', 'human', 'default', 'hand-crafted']
example_name=example_3
example_path=$dst_videos_path/$example_name
mkdir -p $example_path
cp $src_videos_path/human/example-3-human-skade-vs-4glory.mp4 $example_path/${example_name}_${human_name}.mp4
cp $src_videos_path/learned/example-3-learned.mp4 $example_path/${example_name}_${learned_name}.mp4
cp $src_videos_path/default/example-3-default.mp4 $example_path/${example_name}_${default_name}.mp4
cp $src_videos_path/handcrafted/example-3-handcrafted.mp4 $example_path/${example_name}_${handcrafted_name}.mp4

#['default', 'hand-crafted', 'learned', 'human']
example_name=example_4
example_path=$dst_videos_path/$example_name
mkdir -p $example_path
cp $src_videos_path/human/example-4-human-goodjob-vs-websterz.mp4 $example_path/${example_name}_${human_name}.mp4
cp $src_videos_path/learned/example-4-learned.mp4 $example_path/${example_name}_${learned_name}.mp4
cp $src_videos_path/default/example-4-default.mp4 $example_path/${example_name}_${default_name}.mp4
cp $src_videos_path/handcrafted/example-4-handcrafted.mp4 $example_path/${example_name}_${handcrafted_name}.mp4

#['human', 'default', 'learned', 'hand-crafted']
example_name=example_5
example_path=$dst_videos_path/$example_name
mkdir -p $example_path
cp $src_videos_path/human/example-5-human-finest-vs-4glory.mp4 $example_path/${example_name}_${human_name}.mp4
cp $src_videos_path/learned/example-5-learned.mp4 $example_path/${example_name}_${learned_name}.mp4
cp $src_videos_path/default/example-5-default.mp4 $example_path/${example_name}_${default_name}.mp4
cp $src_videos_path/handcrafted/example-5-handcrafted.mp4 $example_path/${example_name}_${handcrafted_name}.mp4

#['default', 'hand-crafted', 'human', 'learned']
example_name=example_6
example_path=$dst_videos_path/$example_name
mkdir -p $example_path
cp $src_videos_path/human/example-6-human-1win-vs-777.mp4 $example_path/${example_name}_${human_name}.mp4
cp $src_videos_path/learned/example-6-learned.mp4 $example_path/${example_name}_${learned_name}.mp4
cp $src_videos_path/default/example-6-default.mp4 $example_path/${example_name}_${default_name}.mp4
cp $src_videos_path/handcrafted/example-6-handcrafted.mp4 $example_path/${example_name}_${handcrafted_name}.mp4

#['learned', 'human', 'hand-crafted', 'default']
example_name=example_7
example_path=$dst_videos_path/$example_name
mkdir -p $example_path
cp $src_videos_path/human/example-7-human-nip-vs-entropiq.mp4 $example_path/${example_name}_${human_name}.mp4
cp $src_videos_path/learned/example-7-learned.mp4 $example_path/${example_name}_${learned_name}.mp4
cp $src_videos_path/default/example-7-default.mp4 $example_path/${example_name}_${default_name}.mp4
cp $src_videos_path/handcrafted/example-7-handcrafted.mp4 $example_path/${example_name}_${handcrafted_name}.mp4

#['hand-crafted', 'learned', 'human', 'default']
example_name=example_8
example_path=$dst_videos_path/$example_name
mkdir -p $example_path
cp $src_videos_path/human/example-8-human-g2-vs-complexity.mp4 $example_path/${example_name}_${human_name}.mp4
cp $src_videos_path/learned/example-8-learned.mp4 $example_path/${example_name}_${learned_name}.mp4
cp $src_videos_path/default/example-8-default.mp4 $example_path/${example_name}_${default_name}.mp4
cp $src_videos_path/handcrafted/example-8-handcrafted.mp4 $example_path/${example_name}_${handcrafted_name}.mp4
