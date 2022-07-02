import cv2
import numpy as np

from argparse import ArgumentParser
from pathlib  import Path


if __name__ == '__main__':
    parser = ArgumentParser(description='Interactive visualization of CALVIN dataset')
    #parser.add_argument('path', type=str, help='Path to dir containing scene_info.npy')
    #parser.add_argument('-d', '--data', nargs='*', default=['rgb_static', 'rgb_gripper'], help='Data to visualize')
    args = parser.parse_args()
    args.path = '/home/ibrahimm/Documents/dl_lab/calvin/dataset/calvin_debug_dataset/training'
    args.path = '/home/ibrahimm/Documents/dl_lab/calvin/gti_demos/'
    args.data = ['rgb_static', 'rgb_gripper']
    if not Path(args.path).is_dir():
        print(f'Path {args.path} is either not a directory, or does not exist.')
        exit()

    #indices = next(iter(np.load(f'{args.path}/scene_info.npy', allow_pickle=True).item().values()))
    #indices = [0,3716]
    indices = [141,3716] #filtered gti_demos
    #indices = [3717,7433] #filtered gti_demos
    indices = list(range(indices[0], indices[1] + 1))
    
    #annotations = np.load(f'{args.path}/lang_annotations/auto_lang_ann.npy', allow_pickle=True).item()
    #annotations = list(zip(annotations['info']['indx'], annotations['language']['ann']))

    idx = 0
    #ann_idx = -1

    while True:
        t = np.load(f'{args.path}/episode_{indices[idx]:07d}.npz', allow_pickle=True)

        for d in args.data:
            if d not in t:
                print(f'Data {d} cannot be found in transition')
                continue

            cv2.imshow(d, t[d][:,:,::-1])
        #for n, ((low, high), ann) in enumerate(annotations):
        #    if indices[idx] >= low and indices[idx] <= high:
        #        if n != ann_idx:
        #            print(f'{ann}')
        #            ann_idx = n
        print(f"episode_{indices[idx]:07d}.npz")
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == 83:  # Right arrow
            idx = (idx + 1) % len(indices)
        elif key == 81:  # Left arrow
            idx = (len(indices) + idx - 1) % len(indices)
        else:
            print(f'Unrecognized keycode "{key}"')
        