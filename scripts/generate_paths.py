import pickle
import os
import argparse
import random
import math
import json

import numpy as np
import networkx as nx

import habitat_sim
import habitat_sim.bindings as hsim
from habitat_sim.utils.common import quat_from_angle_axis

from itertools import permutations
import gzip

def load_metadata(parent_folder):
    points_file = os.path.join(parent_folder, 'points.txt')
    if "replica" in parent_folder:
        graph_file = os.path.join(parent_folder, 'graph.pkl')
        points_data = np.loadtxt(points_file, delimiter="\t")
        points = list(zip(
            points_data[:, 1],
            points_data[:, 3] - 1.5528907,
            -points_data[:, 2])
        )
    else:
        graph_file = os.path.join(parent_folder, 'graph.pkl')
        points_data = np.loadtxt(points_file, delimiter="\t")
        points = list(zip(
            points_data[:, 1],
            points_data[:, 3] - 1.5,
            -points_data[:, 2])
        )
    if not os.path.exists(graph_file):
        raise FileExistsError(graph_file + ' does not exist!')
    else:
        with open(graph_file, 'rb') as fo:
            graph = pickle.load(fo)

    return points, graph


def generate_graph(points, pathfinder):
    navigable_idx = [i for i, p in enumerate(points) if pathfinder.is_navigable(p)]
    graph = nx.Graph()
    for idx in navigable_idx:
        graph.add_node(idx, point=points[idx])
    for a_idx, a_loc in enumerate(points):
        if a_idx not in navigable_idx:
            continue
        for b_idx, b_loc in enumerate(points):
            if b_idx not in navigable_idx:
                continue
            if a_idx == b_idx:
                continue
            euclidean_distance = np.linalg.norm(np.array(a_loc) - np.array(b_loc))
            if 0.1 < euclidean_distance < 1.01:
                path = habitat_sim.ShortestPath()
                path.requested_start = np.array(a_loc, dtype=np.float32)
                path.requested_end = np.array(b_loc, dtype=np.float32)
                pathfinder.find_path(path)
                # relax the constraint a bit
                if path.geodesic_distance < 1.3:
                    graph.add_edge(a_idx, b_idx)
    return graph



scene = "17DRP5sb8fy"

navmesh_file = "data/scene_datasets/mp3d/{}/{}.navmesh".format(scene, scene)
metadata_folder = os.path.join('data/metadata/mp3d')

scene_metadata_folder = os.path.join(metadata_folder, scene)

pathfinder = hsim.PathFinder()
pathfinder.load_nav_mesh(navmesh_file)
points, _ = load_metadata(scene_metadata_folder)

graph = generate_graph(points, pathfinder)
all_nodes = graph.nodes()
all_nodes_list = list(all_nodes)

permutations_of_nodes = list(permutations(all_nodes_list, 2))

data = {"episodes": []}

for idx, perm_node in enumerate(permutations_of_nodes):
    source_node, dest_node = perm_node
    source_xyz = all_nodes[source_node]["point"]
    dest_xyz = all_nodes[dest_node]["point"]

    heading = random.choice([0, 90, 180, 270])
    heading_radians = math.radians(heading)
    rotation = quat_from_angle_axis(heading_radians, np.array([0, -1, 0]))

    data_instance = {
        "episode_id": str(idx),
        # "scene_id": f"data/scene_datasets/mp3d/{scene}/{scene}.glb",
        "scene_id": f"{scene}/{scene}.glb",
        "start_position": list(source_xyz),
        "start_rotation": [rotation.x, rotation.y, rotation.z, rotation.w],
        "info": {"sound": "infinitely"},
        "goals": [
            {
                "position": list(dest_xyz),
                "radius": 1e-05,
            }
        ],
    }
    data["episodes"].append(data_instance)

json_str = json.dumps(data)                      # 2. string (i.e. JSON)
json_bytes = json_str.encode('utf-8')            # 3. bytes (i.e. UTF-8)

with gzip.open("episodes_creation.json.gz", 'w') as fout:       # 4. fewer bytes (i.e. gzip)
    fout.write(json_bytes)   

# Using a JSON string
with open('episodes_creation.json', 'w') as outfile:
    outfile.write(json_str)
