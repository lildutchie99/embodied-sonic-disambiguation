import csv
import json
import random
import sys
import numpy as np
import os
from scipy.signal import convolve, resample
from scipy.io.wavfile import read as wav_read, write as wav_write

MP3D_SCENE_NAME = '17DRP5sb8fy'

soundspaces_data_directory = 'soundspaces_data'
matterport_data_directory = 'matterport_data'
sounds_directory = 'sounds'
convolved_sounds_directory = 'convolved_sounds'

CAUSOR_MAX_DIST = 5 #meters
NUM_SCENARIOS_TO_GENERATE = 1 #10 #PER TRAJECTORY

################
#TODO-----------------------

#trajs = [[((-10.3, -1.73, 0), set()), ((-9.3, .26, 0), set())]] #x, y, theta
with open(sys.argv[1]) as f:
    trajs_json = json.load(f)
    
trajs = {}
for trajid, trajdata in trajs_json.items():
    trajs[trajid] = [((trj['position'][0], trj['position'][1], trj['heading']), set(trj['object_ids'])) for trj in trajdata]
#TODO-------------------
################

if not os.path.exists(convolved_sounds_directory): os.mkdir(convolved_sounds_directory)

def eucdist(p1, p2): return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_scene_point_idx(scene_points, pos):
    bestdist = np.inf
    bestidx = -1
    for i, (px, py) in enumerate(scene_points):
        d = abs(pos[0] - px) + abs(pos[1] - py) #np.sqrt((pos[0] - px)**2 + (pos[1] - py)**2)
        if d < bestdist:
            bestdist = d
            bestidx = i
    assert bestidx != -1, 'ERROR: NO MATCHING POINT IN SCENE FOR POS'
    #if bestdist > 0.5:
    #    print('note: bestdist is', bestdist, 'for pos ', pos)
    return bestidx

category_names = {}
category_names[0] = 'NULL'
category_names[-1] = 'NULL'
with open('Matterport/metadata/category_mapping.tsv') as f:
    r = csv.reader(f, delimiter='\t')
    next(r)
    for line in r:
        category_names[int(line[0])] = line[1]
        
with open('category_to_sound.json') as f:
    cat2sound = json.load(f)

sound2instances = {}
obj_index_to_pos = {}
category_index_to_category_mapping_index = {-1: -1}
with open(os.path.join(matterport_data_directory, MP3D_SCENE_NAME, 'house_segmentations', '%s.house' % MP3D_SCENE_NAME)) as f:
    for l in f:
        line = l.split()

        if line[0] == 'O':
            obj_index_to_pos[int(line[1])] = (float(line[4]), float(line[5]), float(line[6]))
            
            catname = category_names[category_index_to_category_mapping_index[int(line[3])]]
            if catname not in cat2sound: continue #not annotated/not an object associated with a sound
            soundname = cat2sound[catname]
            
            if soundname not in sound2instances: sound2instances[soundname] = []
            sound2instances[soundname].append((int(line[1]), catname)) #line
        elif line[0] == 'C': #category mappings
            category_index_to_category_mapping_index[int(line[1])] = int(line[2])

scene_points = []
with open(os.path.join(soundspaces_data_directory, 'metadata/mp3d', MP3D_SCENE_NAME, 'points.txt')) as f:
    for line in f:
        _, x, y, _ = line.split()
        scene_points.append((float(x), float(y)))

configs = {}
for trajid, traj in trajs.items():
    visible_objects_for_whole_trajectory = set()
    for _, v in traj: visible_objects_for_whole_trajectory |= v
            
    #print(sound2instances)

    s2i_filt = {k: v for k, v in sound2instances.items() if len(v) >= 2}

    sound_obj_pairs = [(snd, causor) for snd, causors in s2i_filt.items() for causor in causors]
    random.shuffle(sound_obj_pairs)

    configs_for_traj = []
    for snd, causor in sound_obj_pairs:
        causorid, causorname = causor
        if causorid not in visible_objects_for_whole_trajectory: continue #it has to be visible at some point
        
        causor_x, causor_y = obj_index_to_pos[causorid][:2]
        
        distractors = set(s2i_filt[snd]) - {causor,}
        
        candidate_frames_for_sound_to_play = [i for i, (pos, visible_objs) in enumerate(traj) \
            #look at frames where object not visible
            if causorid not in visible_objs \
            
            #and agent is reasonably close to object
            and eucdist(pos, (causor_x, causor_y)) < CAUSOR_MAX_DIST \
            
            #and there is some distractor object which could also be the correct answer (an object which makes the relevant sound and is not visible at this point)
            and any([eucdist(pos, obj_index_to_pos[disid]) < CAUSOR_MAX_DIST for disid, _ in (distractors - visible_objs)])]
        
        if candidate_frames_for_sound_to_play:
            timestep_to_play_sound = random.choice(candidate_frames_for_sound_to_play)
            agent_x, agent_y, agent_theta = traj[timestep_to_play_sound][0] #pos of agent at time of sound
            
            agent_pos_idx = get_scene_point_idx(scene_points, (agent_x, agent_y))
            causor_pos_idx = get_scene_point_idx(scene_points, (causor_x, causor_y))
            
            best_theta = -1
            best_diff = np.inf
            for candidate_theta in os.listdir(os.path.join(soundspaces_data_directory, 'binaural_rirs/mp3d', MP3D_SCENE_NAME)):
                try:
                    candidate_theta_float = float(candidate_theta)
                except: continue
                
                angdiff = 180 - abs(abs(candidate_theta_float - agent_theta) - 180)
                if angdiff < best_diff:
                    best_theta = candidate_theta
                    best_diff = angdiff
            assert best_theta != -1 #no idea how this could fail but just to be safe
            
            dry_sample_rate, dry_data = wav_read(os.path.join(sounds_directory, snd + '.wav'))
            
            ir_path = os.path.join(soundspaces_data_directory, 'binaural_rirs/mp3d/%s/%s/%d-%d.wav' % (MP3D_SCENE_NAME, best_theta, agent_pos_idx, causor_pos_idx))
            
            #uncomment for dry run using identity convolution
            #print('reading %s' % ir_path)
            #ir_sample_rate, ir_data = 16000, np.array(([[1, 1]] + [[0, 0]]*15999), dtype=np.float64)
            ir_sample_rate, ir_data = wav_read(ir_path)
            
            assert dry_data.shape[0] == dry_sample_rate, 'wrong size input data, should be exactly one second'
            assert ir_sample_rate == 16000 #I hope?
            
            dry_rs = resample(dry_data, 16000) #get into 16k
            dry_rs = dry_rs.astype(np.float64) / dry_rs.max() #normalize to (-1, 1)
            if dry_rs.shape[-1] == 1: #make it binaural
                dry_rs = np.repeat(dry_rs, 2, axis=-1)
            elif len(dry_rs.shape) == 1:
                dry_rs = np.repeat(dry_rs[:,None], 2, axis=-1)
                
            #print('dry rs shape', dry_rs.shape)
            #print('ir shape', ir_data.shape)
            
            wet = convolve(dry_rs, ir_data, mode='same') #perform convolution
            
            wav_write(os.path.join(convolved_sounds_directory, 'sound_%s_%d.wav' % (trajid, len(configs_for_traj))), 16000, wet)
            
            configs_for_traj.append({'causorid': causorid, 'causorname': causorname, 'causorsound': snd, 'timestep_to_play_sound': timestep_to_play_sound,
            'gps': [x[0] for x in traj]})
            
            if len(configs_for_traj) >= NUM_SCENARIOS_TO_GENERATE: break
            
    configs[trajid] = configs_for_traj
    #print('Generated %d scenarios' % len(res))
    #print(res)
    
with open('scenarios_%s.json' % MP3D_SCENE_NAME, 'w') as f:
    json.dump(configs, f)
