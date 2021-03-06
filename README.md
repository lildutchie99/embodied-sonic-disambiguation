# embodied-sonic-disambiguation
WN 2022 EECS 692 Project at University of Michigan.

## Object-Level Sound Localization in Embodied Environments

Usage instructions:
- Gather sounds correlating to each value in category_to_sound.json and place them in a folder in the root directory named `sounds`
- Clone the Matterport3D repository into the root directory (https://github.com/niessner/Matterport)
- Download Matterport3D house segmentations into a folder named `matterport_data`. The directory structure should be as follows: matterport_data/<Matterport environment ID>/house_segmentations/<Matterport environment ID>.house. See Matterport3D documentation for instructions on requesting and accessing this data.
- Create a folder named `soundspaces_data`, and download and place the `binaural_rirs` and `metadata` folders from the SoundSpaces dataset there. See SoundSpaces documentation for further information on downloading.
- Run `scripts/generate_paths.py` to create episodes which contain start and end positions for trajectories. Then, run `scripts/get_trajectories.py` which computes shortest paths from start to end positions and save them to `traj_data.json`.
- Invoke `scenario_gen.py` with the trajectory JSON as an argument to generate the convolved sounds (stored in `convolved_sounds`) and the full dataset JSON (scenarios_<Matterport environment ID>.json).
- Invoke the `split_dataset` function from trainer.py to split the full dataset JSON into train, validation, and test JSONS.
- Invoke `train_gpu.py` with command line options as documented in the script's help menu.

## References
Initial model code duplicated from [facebookresearch/sound-spaces](https://github.com/facebookresearch/sound-spaces/tree/main/ss_baselines/av_nav), under [Creative Commons 4.0 License](https://github.com/facebookresearch/sound-spaces/blob/main/LICENSE). For more information, please see the following page: https://vision.cs.utexas.edu/projects/audio_visual_navigation/

