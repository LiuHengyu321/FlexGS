#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#


import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, select_render, prune_list, gumbel_select_render, gumbel_select_render_infer
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
# import time

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_set(
    model_path, name, iteration, views, 
    gaussians, pipeline, background, cam_type, 
    ratio=1, selected_mask=None
):
    ratio_name = str(ratio)
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), f"renders_{ratio_name}")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    print("point nums:",gaussians._xyz.shape[0])
    
    stage_type = "fine"
    if ratio == 1:
        stage_type = "coarse"
        ratio = 1
    
    important_score_list = []
    selected_mask_list = []
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        
        rendered_name = "render"
        if stage_type == "fine":
            rendered_name = "selected_render"
        
        render_pkg = select_render(
            view, gaussians, pipeline, background, 
            cam_type=cam_type, stage=stage_type, 
            ratio=ratio, selected_mask=selected_mask
        )
        rendering = render_pkg["render"]
        
        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt = view['image'].cuda()
            gt_list.append(gt)

    time2=time()
    
    print("FPS:",(len(views)-1)/(time2-time1))
    
    # modified by piang
    multithread_write(gt_list, gts_path)
    multithread_write(render_list, render_path)
    # imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30)
    
def render_sets(
    dataset : ModelParams, 
    hyperparam, 
    iteration : int, 
    pipeline : PipelineParams, 
    skip_train : bool, 
    skip_test : bool, 
    skip_video: bool,
    time_ratios: list,
):
    ply_dir = os.path.join(dataset.model_path, "saved_ply")
    print("ply_dir:", ply_dir)
    os.makedirs(ply_dir, exist_ok=True)
    gaussians = GaussianModel(dataset.sh_degree, hyperparam, is_gumbel=True)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    cam_type = scene.dataset_type
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    means3D = gaussians.get_xyz
    scales = gaussians._scaling
    rotations = gaussians._rotation
    
    scales = gaussians.scaling_activation(scales)
    rotations = gaussians.rotation_activation(rotations)
    
    for ratio in time_ratios:
        with torch.no_grad():
            # selected_mask = gumbel_hard_mask > 0
            # start_time = time.time()
            time_gumbel = torch.tensor(ratio, dtype=torch.float32).to(means3D.device).repeat(means3D.shape[0], 1)
            gumbel_soft_mask, gumbel_hard_mask, soft1 = gaussians.gumbel_net(means3D, rotations, scales, time_gumbel)

            sorted_imp_list, _ = torch.sort(soft1, dim=0)
            index_nth_percentile = int((1 - ratio) * (sorted_imp_list.shape[0] - 1))
            value_nth_percentile = sorted_imp_list[index_nth_percentile]
            selected_mask = torch.tensor(soft1 >= value_nth_percentile, dtype=torch.float32)
            selected_mask1 = torch.tensor(soft1 >= value_nth_percentile, dtype=torch.bool)
            print("Predict ratio:", torch.sum(selected_mask) / torch.sum(torch.ones_like(selected_mask)))
            print("Real ratio:", ratio)
            
            ply_name = ply_dir + "/iteration_" + str(iteration) +"/prune_" + str(ratio) + ".ply"
            gaussians.save_select(path=ply_name, selected_mask=selected_mask1, ratio=ratio)
            # ply_name = os.path.join(ply_dir, "iteration_" + str(iteration) +"prune_" + str(ratio) + ".ply")
            # gaussians.save_select(path=ply_name, selected_mask=selected_mask)
            selected_mask = selected_mask > 0
            
            # end_time = time.time()
            
            # print("Process seconds:", end_time - start_time)
            if not skip_train:
                render_set(
                    dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), 
                    gaussians, pipeline, background, cam_type, 
                    ratio=ratio, selected_mask=selected_mask
                )
                
            if not skip_test:
                render_set(
                    dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), 
                    gaussians, pipeline, background, cam_type, 
                    ratio=ratio, selected_mask=selected_mask
                )
            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    # parser.add_argument("--time_ratios", nargs="+", type=float, default=[1, 0.90])
    parser.add_argument("--time_ratios", nargs="+", type=float, default=[0.01, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10])
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video,
        args.time_ratios
    )

