import pydot as dot

g = dot.Dot(graph_type='digraph')
g.set_rankdir("TB")
g.set("newrank", "True")

parent_cluster = dot.Cluster("hi",  rank="same")
g.add_subgraph(parent_cluster)
parent_cluster.add_node(dot.Node("t0_parent", label="t0_parent"))
parent_cluster.add_node(dot.Node("t1_parent", label="t1_parent"))
parent_cluster.add_edge(dot.Edge("t0_parent", "t1_parent"))
g.add_node(dot.Node("t0_child", label="t0_child"))
g.add_node(dot.Node("t1_child", label="t1_child"))
g.add_edge(dot.Edge("t0_parent", "t0_child"))
g.add_edge(dot.Edge("t1_parent", "t1_child"))

far_cluster = dot.Cluster("bye", rank="same", rankdir="LR")
g.add_subgraph(far_cluster)
far_cluster.add_node(dot.Node("t0_child_2", label="t0_child_2"))
far_cluster.add_node(dot.Node("t0_child_3", label="t0_child_3"))
g.add_edge(dot.Edge("t0_child", "t0_child_2"))
far_cluster.add_edge(dot.Edge("t0_child_2", "t0_child_3"))

other_far_cluster = dot.Cluster("bye2", rank="same", rankdir="LR")
g.add_subgraph(other_far_cluster)
other_far_cluster.add_node(dot.Node("t1_child_2", label="t1_child_2"))
other_far_cluster.add_node(dot.Node("t1_child_3", label="t1_child_3"))
g.add_edge(dot.Edge("t1_child", "t1_child_2"))
other_far_cluster.add_edge(dot.Edge("t1_child_2", "t1_child_3"))

g.add_edge(dot.Edge("t0_child_3", "t1_child_2", contraint="False", dir="None"))
print(g)
#s0 = """
#digraph g {
#    rankdir="TB";
#
#    subgraph hi {
#        rank=same;
#        t0_parent;
#        t1_parent;
#        t0_parent -> t1_parent;
#    }
#}
#"""
#g = dot.graph_from_dot_data(s0)[0]
g.write_png("models/tmp.png")

s = """
digraph g {
    rankdir="TB";

    subgraph hi { rank=same;
        0;
        01;
        0 -> 01;
    }
}
"""
sg = dot.graph_from_dot_data(s)[0]

sg.write_png("models/tmp2.png")
