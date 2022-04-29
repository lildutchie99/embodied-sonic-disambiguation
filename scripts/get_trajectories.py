import os
import shutil
import math
import json
import tqdm
from PIL import Image

import numpy as np

import habitat
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
from matterport_label_to_color_map import hex2rgb, mpcat40index2hex, get_random_hex_color

from habitat_sim.utils.common import quat_to_angle_axis

cv2 = try_cv2_import()

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)


additional_hex_dict = {}

def render_semantic_mpcat40(im, mapping):
    out = np.zeros(im.shape, dtype=np.int) # class 0 -> void
    out_rgb = np.zeros(im.shape + (3,), dtype=np.uint8)
    out_rgb_instance = np.zeros(im.shape + (3,), dtype=np.uint8)

    object_counts = {i:0 for i in range(42)}

    global additional_hex_dict
    object_ids = np.unique(im)
    for oid in object_ids:
        mpcat40_index = mapping[oid]

        # remap everything void/unlabeled/misc/etc .. to misc class 40
        # (void:0,  unlabeled: 41, misc=40)
        if mpcat40_index <= 0 or mpcat40_index > 40: mpcat40_index = 40 # remap -1 to misc

        new_object_index = object_counts[mpcat40_index] * 42 + mpcat40_index
        if mpcat40_index not in [0, 40]:
            object_counts[mpcat40_index] = object_counts[mpcat40_index] + 1

        color = hex2rgb(mpcat40index2hex[mpcat40_index])
        if new_object_index in mpcat40index2hex:
            hex = mpcat40index2hex[new_object_index]
        else:
            if new_object_index in additional_hex_dict:
                hex = additional_hex_dict[new_object_index]
            else:
                hex = get_random_hex_color()
                additional_hex_dict[new_object_index] = hex
        color_instance = hex2rgb(hex)

        out[im==oid] = new_object_index

        out_rgb[im==oid] = color
        out_rgb_instance[im==oid] = color_instance

    return out, out_rgb, out_rgb_instance

class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )

def save_images(dirname, images, label):
    for idx, im in enumerate(tqdm.tqdm(images)):
        filename = os.path.join(dirname, f"{idx}_{label}.png")
        PIL_im = Image.fromarray(im)
        PIL_im.save(filename)


def shortest_path_example():
    config = habitat.get_config(config_paths="config.yaml")
    config.defrost()
    config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.freeze()
    with SimpleRLEnv(config=config) as env:
        goal_radius = env.episodes[0].goals[0].radius
        if goal_radius is None:
            goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
        follower = ShortestPathFollower(
            env.habitat_env.sim, goal_radius, False
        )

        scene = env.habitat_env.sim.semantic_scene

        instance2object_map = {int(obj.id.split("_")[-1]): obj.category.index(mapping='mpcat40') for obj in scene.objects}
        # instance2object_map = {int(obj.id.split("_")[-1]): obj.category.index(mapping='raw') for obj in scene.objects}

        data_output = {}
        

        print("Environment creation successful")
        for episode in range(2652):
            env.reset()

            episode_id = env.current_episode.episode_id

            dirname = os.path.join(
                IMAGE_DIR, "shortest_path_example", episode_id
            )
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            os.makedirs(dirname)    
            print("Agent stepping around inside environment.")
            timestep = 0
            episode_output = []

            rgb_ims = []
            semantic_ims = []

            unique_oids = []

            while not env.habitat_env.episode_over:
                best_action = follower.get_next_action(
                    env.habitat_env.current_episode.goals[0].position
                )
                if best_action is None:
                    break

                observations, reward, done, info = env.step(best_action)

                rgb_im = observations["rgb"]

                semantic = observations["semantic"]

                unique_oids.extend(np.unique(semantic).tolist())

                semantic_im, semantic_rgb, semantic_rgb_instance = render_semantic_mpcat40(semantic, instance2object_map)

                object_ids = np.unique(semantic).tolist()
                
                agent_state = env.habitat_env.sim.get_agent_state()
                position = agent_state.position.astype(np.float).tolist()
                rotation = agent_state.rotation

                heading = round(math.degrees(quat_to_angle_axis(rotation)[0]))

                # top_down_map = draw_top_down_map(info, rgb_im.shape[0])

                # output_im = np.concatenate((rgb_im, semantic_rgb_instance, top_down_map), axis=1)
                # output_im = np.concatenate((rgb_im, semantic_rgb_instance), axis=1)

                # images.append(output_im)

                rgb_ims.append(rgb_im)
                semantic_ims.append(semantic_rgb_instance)

                time_instance = {
                    "timestep": timestep, 
                    "position": position, 
                    "heading": heading, 
                    "object_ids": object_ids,
                }
                episode_output.append(time_instance)
                timestep += 1

            data_output[episode_id] = episode_output
            save_images(dirname, rgb_ims, "rgb")
            save_images(dirname, semantic_ims, "semantic")

            # images_to_video(images, dirname, "trajectory")
            print(f"Episode {episode} finished")


    with open('traj_data.json', 'w') as outfile:
        json.dump(data_output, outfile) 

shortest_path_example()