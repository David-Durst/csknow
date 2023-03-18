import pydot as dot
from dataclasses import dataclass
from enum import Enum
from typing import List


class DBNType(Enum):
    CausalInput = 0
    LatentState = 1
    ACausalOutput = 2


class DBNNode:
    node_type: DBNType
    name: str
    id: int
    next_id = 0

    def __init__(self, node_type: DBNType, name: str):
        self.node_type = node_type
        self.name = name
        self.id = DBNNode.next_id
        DBNNode.next_id += 1


def reset_dbn_counter():
    DBNNode.next_id = 0


class DBNEdge:
    src: int
    dst: int

    def __init__(self, src: DBNNode, dst: DBNNode):
        self.src = src.id
        self.dst = dst.id


@dataclass
class DBN:
    nodes: List[DBNNode]
    edges: List[DBNEdge]
    temporal_nodes: List[DBNNode]

    def get_nodes(self, node_type: DBNType) -> List[DBNNode]:
        result = []
        for n in self.nodes:
            if n.node_type == node_type:
                result.append(n)
        return result


def plot_dbn(dbn: DBN):
    g = dot.Dot(graph_type='digraph')
    g.set_rankdir("TB")
    g.set("newrank", "True")

    node_to_time_cluster = {}
    node_to_type_cluster = {}
    parent_cluster = dot.Cluster("parent", rank="same")
    for t in [0,1]:
        time_cluster = dot.Cluster(str(t), label=f"Time Slice {t}", fontcolor="#DADADA", bgcolor="#737272",
                                   rank="same", rankdir="LR")
        g.add_subgraph(time_cluster)
        node_types = [DBNType.CausalInput, DBNType.LatentState, DBNType.ACausalOutput]
        for nt in node_types:
            if nt == DBNType.CausalInput:
                label_str = "Causal Input"
                color = "#f78d86"
            elif nt == DBNType.LatentState:
                label_str = "Latent State"
                color = "#91b9fa"
            else:
                label_str = "ACausal Output"
                color = "#8ff2ae"
            print(label_str)
            type_cluster = dot.Cluster(str(t) + label_str, label=label_str, fontcolor="#000000", bgcolor="#DDDDDD",
                                       rank="same", rankdir="same")
            time_cluster.add_subgraph(type_cluster)
            for n in dbn.get_nodes(nt):
                type_cluster.add_node(dot.Node(f"{t}_{n.id}", label=n.name, fillcolor=color, style="filled"))
                node_to_time_cluster[n.id] = time_cluster
                node_to_type_cluster[n.id] = type_cluster

        g.set_edge_defaults(color="blue", constraint="True")
        for e in dbn.edges:
            #should_constrain = "True"
            #if dbn.nodes[e.src].node_type == dbn.nodes[e.dst].node_type:
            #    should_constrain = "False"
            if node_to_type_cluster[e.src] == node_to_type_cluster[e.dst]:
                node_to_type_cluster[e.src].add_edge(dot.Edge(f"{t}_{e.src}", f"{t}_{e.dst}", constraint="False"))
            else:
                node_to_time_cluster[e.src].add_edge(dot.Edge(f"{t}_{e.src}", f"{t}_{e.dst}", constraint="True"))

    g.set_edge_defaults(color="black", constraint="True")
    for n in dbn.temporal_nodes:
        g.add_edge(dot.Edge(f"0_{n.id}", f"1_{n.id}"))

    g.set_edge_defaults(color="invis", constraint="True")
    for n in dbn.nodes:
        g.add_edge(dot.Edge(f"0_{n.id}", f"1_{n.id}", rank="same"))

    g.write_png("models/dbn.png")


visible = DBNNode(DBNType.CausalInput, "Visible")
engage = DBNNode(DBNType.LatentState, "Engage")
target = DBNNode(DBNType.LatentState, "Target")
world_speed = DBNNode(DBNType.ACausalOutput, "World Speed")

plot_dbn(DBN(
    [visible, engage, target, world_speed],
    [DBNEdge(visible, engage), DBNEdge(visible, target), DBNEdge(engage, target), DBNEdge(target, world_speed)],
    [engage, target]
))


