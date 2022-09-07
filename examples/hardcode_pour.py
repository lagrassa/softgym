import os.path as osp
import argparse
import numpy as np
np.random.seed(0)

import torch
import tensorflow

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
import pyflex
from matplotlib import pyplot as plt


def show_depth():
    # render rgb and depth
    img, depth = pyflex.render()
    img = img.reshape((720, 720, 4))[::-1, :, :3]
    depth = depth.reshape((720, 720))[::-1]
    # get foreground mask
    rgb, depth = pyflex.render_cloth()
    depth = depth.reshape(720, 720)[::-1]
    # mask = mask[:, :, 3]
    # depth[mask == 0] = 0
    # show rgb and depth(masked)
    depth[depth > 5] = 0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[1].imshow(depth)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='PourWaterPlant')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    parser.add_argument('--test_depth', type=int, default=0, help='If to test the depth rendering by showing it')

    args = parser.parse_args()

    env_kwargs = env_arg_dict[args.env_name]

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.reset()

    frames = [env.get_image(args.img_size, args.img_size)]
    point_dataset = None
    vector_dataset = None
    actions = []
    low_pour = True
    #for i in range(100000):
    #    pyflex.render()
    if low_pour:
        for i in range(4):
            actions.append(np.array([0.0,-0.07, 0]))
        for i in range(7):
            actions.append(np.array([0.05,0, 0]))
        for i in range(2):
            actions.append(np.array([0.02,0, 0]))
        for i in range(8):
            actions.append(np.array([0, 0.0, 0.4]))
        for i in range(14):
            actions.append(np.array([0, 0.0, 0.02]))
    if not low_pour:
        for i in range(3):
            actions.append(np.array([0.00, 0.10, 0]))
        for i in range(4):
            actions.append(np.array([0.06, 0.0, 0]))
        for i in range(8):
            actions.append(np.array([0, 0.0, 0.4]))
        for i in range(14):
            actions.append(np.array([0, 0.0, 0.02]))
    env.horizon = len(actions)
    for i in range(env.horizon):
        action = actions[i]
        # By default, the environments will apply action repitition. The option of record_continuous_video provides rendering of all
        # intermediate frames. Only use this option for visualization as it increases computation.
        data = env.step(action, record_continuous_video=True, img_size=args.img_size)
        if point_dataset is None:
            vector_dataset = np.zeros((env.horizon, 13))
            point_dataset = np.zeros((env.horizon,data[0][1].shape[0], 4))
        _, _, _, info = data
        #dataset.append(data[0])
        #point_dataset[i, :]  = data[0][1]
        vector_dataset[i]  = data[0][0]
        #frames.extend(info['flex_env_recorded_frames'])
        if args.test_depth:
            show_depth()
    np.save("data/pass_water_vector_dataset.npy", vector_dataset)
    np.save("data/pass_water_point_dataset.npy", point_dataset)
    if args.save_video_dir is not None:
        save_name = osp.join(args.save_video_dir, args.env_name + '.gif')
        save_numpy_as_gif(np.array(frames), save_name)
        print('Video generated and save to {}'.format(save_name))


if __name__ == '__main__':
    main()
