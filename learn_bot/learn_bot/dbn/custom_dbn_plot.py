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


def plot_dbn(dbn: DBN):
    g = dot.Dot(graph_type='digraph')
    g.set_rankdir("LR")
    g.set_splines("ortho")

    for t in [0,1]:
        cluster = dot.Cluster(str(t), label=f"Time Slice {t}", bgcolor="#DDDDDD", rankdir="same")
        g.add_subgraph(cluster)
        for n in dbn.nodes:
            if n.node_type == DBNType.CausalInput:
                color = "#f78d86"
            elif n.node_type == DBNType.LatentState:
                color = "#91b9fa"
            else:
                color = "#8ff2ae"
            cluster.add_node(dot.Node(f"{t}_{n.id}", label=n.name, fillcolor=color, style="filled"))

        g.set_edge_defaults(color="blue", constraint="False")
        for e in dbn.edges:
            g.add_edge(dot.Edge(f"{t}_{e.src}", f"{t}_{e.dst}"))

    g.set_edge_defaults(color="black", constraint="True")
    for n in dbn.temporal_nodes:
        g.add_edge(dot.Edge(f"0_{n.id}", f"1_{n.id}"))

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



