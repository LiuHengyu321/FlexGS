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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time
import gc


def count_render(
    viewpoint_camera, 
    pc : GaussianModel, 
    pipe, 
    bg_color : torch.Tensor, 
    scaling_modifier = 1.0, 
    override_color = None, 
    # stage="fine", 
    cam_type=None,
    # ratio=0.0
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug,
            # f_count=True,
            f_count=True,
        )
    else:
        raster_settings = viewpoint_camera['camera']
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table

    means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations,opacity, shs


    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass 
    else:
        colors_precomp = override_color

    gaussians_count, important_score, rendered_image, radii = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,    # None
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp       # None
    )
    
    return {
        "render": rendered_image,
        "visibility_filter" : radii > 0,
        "radii": radii,
        "viewspace_points": screenspace_points,
        "gaussians_count": gaussians_count,
        "important_score": important_score,
    }


def render(
    viewpoint_camera, 
    pc : GaussianModel, 
    pipe, 
    bg_color : torch.Tensor, 
    scaling_modifier = 1.0, 
    override_color = None, 
    stage="fine", 
    cam_type=None,
    ratio=0.0
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    important_score = None
    if stage != "coarse":
        important_score = count_render(
            viewpoint_camera,
            pc,
            pipe,
            bg_color,
            scaling_modifier,
            override_color,
            cam_type
        )["important_score"]
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug,
            # f_count=True,
            f_count=False,
        )
        # time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        # time = torch.tensor(ratio, dtype=torch.float32, requires_grad=True).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        # time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        # time = torch.tensor(ratio, requires_grad=True).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table

    # if stage == "coarse":
    #     means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations,opacity, shs
        
    # else:
    #     means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
    #         means3D, scales, 
    #         rotations, opacity, shs,
    #         time
    #     )

    means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations,opacity, shs
    
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
    else:
        colors_precomp = override_color

    full_rendered_image, full_radii = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,    # None
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp       # None
    )
    
    selected_rendered_image = None
    selected_radii = None
    selected_visibity = None
        
    if important_score is not None:
        sorted_tensor, _ = torch.sort(important_score, dim=0)
        index_nth_percentile = int(ratio * (sorted_tensor.shape[0] - 1))
        value_nth_percentile = sorted_tensor[index_nth_percentile]
        selected_mask = important_score > value_nth_percentile
        
        select_means3D = means3D_final[selected_mask]
        select_means2D = means2D[selected_mask]
        select_shs = shs_final[selected_mask]
        select_opacity = opacity[selected_mask]
        select_scales = scales_final[selected_mask]
        select_rotations = rotations_final[selected_mask]
        
        time = torch.tensor(ratio, dtype=torch.float32, requires_grad=True).to(select_means3D.device).repeat(select_means3D.shape[0],1)
        # print("ratio:", ratio)
        select_means3D, select_scales, select_rotations, select_opacity, select_shs = pc._deformation(
            select_means3D, select_scales, 
            select_rotations, select_opacity, select_shs,
            time
        )
        
        selected_rendered_image, selected_radii = rasterizer(
            means3D = select_means3D,
            means2D = select_means2D,
            shs = select_shs,
            colors_precomp = colors_precomp,    # None
            opacities = select_opacity,
            scales = select_scales,
            rotations = select_rotations,
            cov3D_precomp = cov3D_precomp       # None
        )
        selected_visibity = selected_radii > 0

    return {
        "render": full_rendered_image,
        "visibility_filter" : full_radii > 0,
        "radii": full_radii,
        
        "viewspace_points": screenspace_points,
        
        "selected_render": selected_rendered_image,
        "selected_visibility_filter" : selected_visibity,
        "selected_radii": selected_visibity,
        
        "selected_mask": selected_mask,
        "important_score": important_score,
        # "gaussians_count": gaussians_count,
        
    }


def prune_list(gaussians, scene, pipe, background):
    viewpoint_stack = scene.getTrainCameras().copy()
    gaussian_list, imp_list = None, None
    viewpoint_cam = viewpoint_stack.pop()
    
    render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
    gaussian_list, imp_list = (
        render_pkg["gaussians_count"],
        render_pkg["important_score"],
    )
    
    for iteration in range(len(viewpoint_stack)):
        viewpoint_cam = viewpoint_stack.pop()
        render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
        gaussians_count, important_score = (
            render_pkg["gaussians_count"].detach(),
            render_pkg["important_score"].detach(),
        )
        gaussian_list += gaussians_count
        imp_list += important_score
        gc.collect()
    return gaussian_list, imp_list


def select_render(
    viewpoint_camera, 
    pc : GaussianModel, 
    pipe, 
    bg_color : torch.Tensor, 
    scaling_modifier = 1.0, 
    override_color = None, 
    stage="fine", 
    cam_type=None,
    selected_mask=None, 
    ratio=0.0
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug,
            # f_count=True,
            f_count=False,
        )
        # time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        # time = torch.tensor(ratio, dtype=torch.float32, requires_grad=True).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        # time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        # time = torch.tensor(ratio, requires_grad=True).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table

    # if stage == "coarse":
    #     means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations,opacity, shs
        
    # else:
    #     means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
    #         means3D, scales, 
    #         rotations, opacity, shs,
    #         time
    #     )

    means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations,opacity, shs
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
    else:
        colors_precomp = override_color

    full_rendered_image, full_radii = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp,    # None
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp       # None
        )
    
    select_means3D = means3D_final[selected_mask]
    select_means2D = means2D[selected_mask]
    select_shs = shs_final[selected_mask]
    select_opacity = opacity[selected_mask]
    select_scales = scales_final[selected_mask]
    select_rotations = rotations_final[selected_mask]
    
    selected_rendered_image, selected_radii = full_rendered_image, full_radii
    
    if ratio != 1: 
        time = torch.tensor(ratio, dtype=torch.float32, requires_grad=True).to(select_means3D.device).repeat(select_means3D.shape[0],1)
        # print("ratio:", ratio)
        select_means3D, select_scales, select_rotations, select_opacity, select_shs = pc._deformation(
            select_means3D, select_scales, 
            select_rotations, select_opacity, select_shs,
            time
        )
    
        selected_rendered_image, selected_radii = rasterizer(
            means3D = select_means3D,
            means2D = select_means2D,
            shs = select_shs,
            colors_precomp = colors_precomp,    # None
            opacities = select_opacity,
            scales = select_scales,
            rotations = select_rotations,
            cov3D_precomp = cov3D_precomp       # None
        )
    

    return {
        "render": selected_rendered_image,
        "visibility_filter" : selected_radii > 0,
        "radii": selected_radii,
        
        "full_render": full_rendered_image,
        "full_radii": full_radii,
        
        "viewspace_points": screenspace_points,
        # "gaussians_count": gaussians_count,
        
    }
    

def gumbel_select_render(
    viewpoint_camera, 
    pc : GaussianModel, 
    pipe, 
    bg_color : torch.Tensor, 
    scaling_modifier = 1.0, 
    override_color = None, 
    stage="fine", 
    cam_type=None,
    ratio=1.0
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug,
            # f_count=True,
            f_count=False,
        )
        # time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        # time = torch.tensor(ratio, dtype=torch.float32, requires_grad=True).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        # time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        # time = torch.tensor(ratio, requires_grad=True).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table

    # if stage == "coarse":
    #     means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations,opacity, shs
        
    # else:
    #     means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
    #         means3D, scales, 
    #         rotations, opacity, shs,
    #         time
    #     )

    means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations,opacity, shs
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
    else:
        colors_precomp = override_color

    full_rendered_image, full_radii = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp,    # None
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp       # None
        )
    
    selected_rendered_image, selected_radii = full_rendered_image, full_radii
    gumbel_soft_mask, gumbel_hard_mask = None, None
    if ratio != 1: 
        # time_gumbel = torch.tensor(ratio, dtype=torch.int32).to(means3D_final.device).repeat(means3D_final.shape[0], 1)
        time_gumbel = torch.tensor(ratio, dtype=torch.float32).to(means3D_final.device).repeat(means3D_final.shape[0], 1)
        # print("ratio:", ratio)
        gumbel_soft_mask, gumbel_hard_mask, _ = pc.gumbel_net(means3D_final, rotations_final, scales_final, time_gumbel)
        selected_mask_soft = gumbel_hard_mask
        selected_mask_hard = gumbel_hard_mask > 0
        
        print("Pred ratio:", torch.sum(selected_mask_hard) / torch.sum(torch.ones_like(selected_mask_hard)))
        print("Real ratio:", ratio)
        
        select_means3D = means3D_final[selected_mask_hard]
        # select_means2D = means2D[selected_mask]
        select_shs = shs_final[selected_mask_hard]
        select_opacity = opacity[selected_mask_hard]
        select_scales = scales_final[selected_mask_hard]
        select_rotations = rotations_final[selected_mask_hard]
        
        time = torch.tensor(ratio, dtype=torch.float32, requires_grad=True).to(select_means3D.device).repeat(select_means3D.shape[0], 1)
        
        select_means3D, select_scales, select_rotations, _, _ = pc._deformation(
            select_means3D, select_scales, 
            select_rotations, select_opacity, select_shs,
            time
        )
        
        means3D_gumbel = torch.zeros_like(means3D_final)
        scales_gumbel = torch.zeros_like(scales_final)
        rotations_gumbel = torch.zeros_like(rotations_final)

        means3D_gumbel[selected_mask_hard] = select_means3D
        scales_gumbel[selected_mask_hard] = select_scales
        rotations_gumbel[selected_mask_hard] = select_rotations
        
        means3D_gumbel[~selected_mask_hard] = means3D_final[~selected_mask_hard]
        scales_gumbel[~selected_mask_hard] = scales_final[~selected_mask_hard]
        rotations_gumbel[~selected_mask_hard] = rotations_final[~selected_mask_hard]
        
        opacity = opacity * selected_mask_soft.unsqueeze(1)
        
        selected_rendered_image, selected_radii = rasterizer(
            means3D = means3D_gumbel,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp,    # None
            opacities = opacity,
            scales = scales_gumbel,
            rotations = rotations_gumbel,
            cov3D_precomp = cov3D_precomp       # None
        )
    

    return {
        "render": selected_rendered_image,
        "visibility_filter" : selected_radii > 0,
        "radii": selected_radii,
        
        "full_render": full_rendered_image,
        "full_radii": full_radii,
        
        "gumbel_soft_mask": gumbel_soft_mask, 
        "gumbel_hard_mask": gumbel_hard_mask,
        
        "viewspace_points": screenspace_points,
    }

def gumbel_select_render_infer(
    viewpoint_camera, 
    pc : GaussianModel, 
    pipe, 
    bg_color : torch.Tensor, 
    scaling_modifier = 1.0, 
    override_color = None, 
    stage="fine", 
    cam_type=None,
    ratio=1.0
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug,
            # f_count=True,
            f_count=False,
        )
        # time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        # time = torch.tensor(ratio, dtype=torch.float32, requires_grad=True).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        # time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        # time = torch.tensor(ratio, requires_grad=True).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table

    # if stage == "coarse":
    #     means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations,opacity, shs
        
    # else:
    #     means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
    #         means3D, scales, 
    #         rotations, opacity, shs,
    #         time
    #     )

    means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations,opacity, shs
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
    else:
        colors_precomp = override_color

    full_rendered_image, full_radii = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp,    # None
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp       # None
        )
    
    selected_rendered_image, selected_radii = full_rendered_image, full_radii
    gumbel_soft_mask, gumbel_hard_mask = None, None
    if ratio != 1: 
        # time_gumbel = torch.tensor(ratio, dtype=torch.int32).to(means3D_final.device).repeat(means3D_final.shape[0], 1)
        time_gumbel = torch.tensor(ratio, dtype=torch.float32).to(means3D_final.device).repeat(means3D_final.shape[0], 1)
        # print("ratio:", ratio)
        gumbel_soft_mask, gumbel_hard_mask = pc.gumbel_net(means3D_final, rotations_final, scales_final, time_gumbel)
        selected_mask_soft = gumbel_hard_mask
        selected_mask_hard = gumbel_hard_mask > 0
        
        print("Pred ratio:", torch.sum(selected_mask_hard) / torch.sum(torch.ones_like(selected_mask_hard)))
        print("Real ratio:", ratio)
        
        
        select_means3D = means3D_final[selected_mask_hard]
        select_means2D = means2D[selected_mask_hard]
        select_shs = shs_final[selected_mask_hard]
        select_opacity = opacity[selected_mask_hard]
        select_scales = scales_final[selected_mask_hard]
        select_rotations = rotations_final[selected_mask_hard]
        
        time = torch.tensor(ratio, dtype=torch.float32, requires_grad=True).to(select_means3D.device).repeat(select_means3D.shape[0], 1)
        
        select_means3D, select_scales, select_rotations, select_opacity, select_shs = pc._deformation(
            select_means3D, select_scales, 
            select_rotations, select_opacity, select_shs,
            time
        )
        
        # means3D_gumbel = torch.zeros_like(means3D_final)
        # scales_gumbel = torch.zeros_like(scales_final)
        # rotations_gumbel = torch.zeros_like(rotations_final)

        # means3D_gumbel[selected_mask_hard] = select_means3D
        # scales_gumbel[selected_mask_hard] = select_scales
        # rotations_gumbel[selected_mask_hard] = select_rotations
        
        # means3D_gumbel[~selected_mask_hard] = means3D_final[~selected_mask_hard]
        # scales_gumbel[~selected_mask_hard] = scales_final[~selected_mask_hard]
        # rotations_gumbel[~selected_mask_hard] = rotations_final[~selected_mask_hard]
        
        opacity = opacity * selected_mask_soft.unsqueeze(1)
        
        selected_rendered_image, selected_radii = rasterizer(
            means3D = select_means3D,
            means2D = select_means2D,
            shs = select_shs,
            colors_precomp = colors_precomp,    # None
            opacities = select_opacity,
            scales = select_scales,
            rotations = select_rotations,
            cov3D_precomp = cov3D_precomp       # None
        )
    

    return {
        "render": selected_rendered_image,
        "visibility_filter" : selected_radii > 0,
        "radii": selected_radii,
        
        "full_render": full_rendered_image,
        "full_radii": full_radii,
        
        "gumbel_soft_mask": gumbel_soft_mask, 
        "gumbel_hard_mask": gumbel_hard_mask,
        
        "viewspace_points": screenspace_points,
    }
    

def gumbel_select_render_threshold(
    viewpoint_camera, 
    pc : GaussianModel, 
    pipe, 
    bg_color : torch.Tensor, 
    scaling_modifier = 1.0, 
    override_color = None, 
    stage="fine", 
    cam_type=None,
    ratio=1.0,
    cur_tau=1.0,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug,
            # f_count=True,
            f_count=False,
        )
        # time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        # time = torch.tensor(ratio, dtype=torch.float32, requires_grad=True).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        # time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        # time = torch.tensor(ratio, requires_grad=True).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table

    # if stage == "coarse":
    #     means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations,opacity, shs
        
    # else:
    #     means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
    #         means3D, scales, 
    #         rotations, opacity, shs,
    #         time
    #     )

    means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations,opacity, shs
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
    else:
        colors_precomp = override_color

    full_rendered_image, full_radii = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp,    # None
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp       # None
        )
    
    selected_rendered_image, selected_radii = full_rendered_image, full_radii
    gumbel_soft_mask, gumbel_hard_mask = None, None
    if ratio != 1: 
        # time_gumbel = torch.tensor(ratio, dtype=torch.int32).to(means3D_final.device).repeat(means3D_final.shape[0], 1)
        time_gumbel = torch.tensor(ratio, dtype=torch.float32).to(means3D_final.device).repeat(means3D_final.shape[0], 1)
        gumbel_soft_mask, gumbel_hard_mask = pc.gumbel_net(means3D_final, rotations_final, scales_final, time_gumbel, cur_tau=cur_tau)
        
        selected_mask_soft = gumbel_hard_mask
        selected_mask_hard = gumbel_hard_mask > 0
        print("predict ratio:", torch.sum(selected_mask_hard) / torch.sum(torch.ones_like(selected_mask_hard)))
        print("ratio:", ratio)
        
        if (torch.sum(selected_mask_hard) / torch.sum(torch.ones_like(selected_mask_hard)) > 0.20) or (torch.sum(selected_mask_hard) / torch.sum(torch.ones_like(selected_mask_hard)) < 0.0001):
            # print("Mask with soft", torch.sum(selected_mask_hard) / torch.sum(torch.ones_like(selected_mask_hard)).item(), ratio)
            sorted_imp_list, _ = torch.sort(gumbel_soft_mask, dim=0)
            index_nth_percentile = int((1 - ratio) * (sorted_imp_list.shape[0] - 1))
            value_nth_percentile = sorted_imp_list[index_nth_percentile]
            selected_mask_hard = gumbel_soft_mask > value_nth_percentile
        else:
            # print("Mask with hard")
            pass
        # print("Pred ratio:", torch.sum(selected_mask_hard) / torch.sum(torch.ones_like(selected_mask_hard)))
        # print("Real ratio:", ratio)
        
        select_means3D = means3D_final[selected_mask_hard]
        # select_means2D = means2D[selected_mask]
        select_shs = shs_final[selected_mask_hard]
        select_opacity = opacity[selected_mask_hard]
        select_scales = scales_final[selected_mask_hard]
        select_rotations = rotations_final[selected_mask_hard]
        
        time = torch.tensor(ratio, dtype=torch.float32, requires_grad=True).to(select_means3D.device).repeat(select_means3D.shape[0], 1)
        
        select_means3D, select_scales, select_rotations, _, _ = pc._deformation(
            select_means3D, select_scales, 
            select_rotations, select_opacity, select_shs,
            time
        )
        
        means3D_gumbel = torch.zeros_like(means3D_final)
        scales_gumbel = torch.zeros_like(scales_final)
        rotations_gumbel = torch.zeros_like(rotations_final)

        means3D_gumbel[selected_mask_hard] = select_means3D
        scales_gumbel[selected_mask_hard] = select_scales
        rotations_gumbel[selected_mask_hard] = select_rotations
        
        means3D_gumbel[~selected_mask_hard] = means3D_final[~selected_mask_hard]
        scales_gumbel[~selected_mask_hard] = scales_final[~selected_mask_hard]
        rotations_gumbel[~selected_mask_hard] = rotations_final[~selected_mask_hard]
        
        opacity = opacity * selected_mask_soft.unsqueeze(1)
        
        selected_rendered_image, selected_radii = rasterizer(
            means3D = means3D_gumbel,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp,    # None
            opacities = opacity,
            scales = scales_gumbel,
            rotations = rotations_gumbel,
            cov3D_precomp = cov3D_precomp       # None
        )
    

    return {
        "render": selected_rendered_image,
        "visibility_filter" : selected_radii > 0,
        "radii": selected_radii,
        
        "full_render": full_rendered_image,
        "full_radii": full_radii,
        
        "gumbel_soft_mask": gumbel_soft_mask, 
        "gumbel_hard_mask": gumbel_hard_mask,
        
        "viewspace_points": screenspace_points,
    }


def gumbel_select_render_add1(
    viewpoint_camera, 
    pc : GaussianModel, 
    pipe, 
    bg_color : torch.Tensor, 
    scaling_modifier = 1.0, 
    override_color = None, 
    stage="fine", 
    cam_type=None,
    ratio=1.0
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug,
            # f_count=True,
            f_count=False,
        )
        # time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        # time = torch.tensor(ratio, dtype=torch.float32, requires_grad=True).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        # time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        # time = torch.tensor(ratio, requires_grad=True).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table

    # if stage == "coarse":
    #     means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations,opacity, shs
        
    # else:
    #     means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
    #         means3D, scales, 
    #         rotations, opacity, shs,
    #         time
    #     )

    means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations,opacity, shs
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
    else:
        colors_precomp = override_color

    full_rendered_image, full_radii = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp,    # None
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp       # None
        )
    
    selected_rendered_image, selected_radii = full_rendered_image, full_radii
    gumbel_soft_mask, gumbel_hard_mask = None, None
    if ratio != 1: 
        # time_gumbel = torch.tensor(ratio, dtype=torch.int32).to(means3D_final.device).repeat(means3D_final.shape[0], 1)
        time_gumbel = torch.tensor(ratio, dtype=torch.float32).to(means3D_final.device).repeat(means3D_final.shape[0], 1)
        # print("ratio:", ratio)
        gumbel_soft_mask, gumbel_hard_mask = pc.gumbel_net(means3D_final, time_gumbel)
        selected_mask_soft = gumbel_hard_mask
        selected_mask_hard = gumbel_hard_mask > 0
        
        print("Pred ratio:", torch.sum(selected_mask_hard) / torch.sum(torch.ones_like(selected_mask_hard)))
        print("Real ratio:", ratio)
        
        select_means3D = means3D_final[selected_mask_hard]
        # select_means2D = means2D[selected_mask]
        select_shs = shs_final[selected_mask_hard]
        select_opacity = opacity[selected_mask_hard]
        select_scales = scales_final[selected_mask_hard]
        select_rotations = rotations_final[selected_mask_hard]
        
        time = torch.tensor(ratio, dtype=torch.float32, requires_grad=True).to(select_means3D.device).repeat(select_means3D.shape[0], 1)
        
        select_means3D, select_scales, select_rotations, _, _ = pc._deformation(
            select_means3D, select_scales, 
            select_rotations, select_opacity, select_shs,
            time
        )
        
        means3D_gumbel = torch.zeros_like(means3D_final)
        scales_gumbel = torch.zeros_like(scales_final)
        rotations_gumbel = torch.zeros_like(rotations_final)

        means3D_gumbel[selected_mask_hard] = select_means3D
        scales_gumbel[selected_mask_hard] = select_scales
        rotations_gumbel[selected_mask_hard] = select_rotations
        
        means3D_gumbel[~selected_mask_hard] = means3D_final[~selected_mask_hard]
        scales_gumbel[~selected_mask_hard] = scales_final[~selected_mask_hard]
        rotations_gumbel[~selected_mask_hard] = rotations_final[~selected_mask_hard]
        
        opacity = opacity * selected_mask_soft.unsqueeze(1)
        
        selected_rendered_image, selected_radii = rasterizer(
            means3D = means3D_gumbel,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp,    # None
            opacities = opacity,
            scales = scales_gumbel,
            rotations = rotations_gumbel,
            cov3D_precomp = cov3D_precomp       # None
        )
    

    return {
        "render": selected_rendered_image,
        "visibility_filter" : selected_radii > 0,
        "radii": selected_radii,
        
        "full_render": full_rendered_image,
        "full_radii": full_radii,
        
        "gumbel_soft_mask": gumbel_soft_mask, 
        "gumbel_hard_mask": gumbel_hard_mask,
        
        "viewspace_points": screenspace_points,
    }


def gumbel_select_render_ab1(
    viewpoint_camera, 
    pc : GaussianModel, 
    pipe, 
    bg_color : torch.Tensor, 
    scaling_modifier = 1.0, 
    override_color = None, 
    stage="fine", 
    cam_type=None,
    ratio=1.0
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug,
            # f_count=True,
            f_count=False,
        )
        # time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        # time = torch.tensor(ratio, dtype=torch.float32, requires_grad=True).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        # time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        # time = torch.tensor(ratio, requires_grad=True).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table

    # if stage == "coarse":
    #     means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations,opacity, shs
        
    # else:
    #     means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
    #         means3D, scales, 
    #         rotations, opacity, shs,
    #         time
    #     )

    means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations,opacity, shs
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
    else:
        colors_precomp = override_color

    full_rendered_image, full_radii = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp,    # None
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp       # None
        )
    
    selected_rendered_image, selected_radii = full_rendered_image, full_radii
    gumbel_soft_mask, gumbel_hard_mask = None, None
    if ratio != 1: 
        # time_gumbel = torch.tensor(ratio, dtype=torch.int32).to(means3D_final.device).repeat(means3D_final.shape[0], 1)
        time_gumbel = torch.tensor(ratio, dtype=torch.float32).to(means3D_final.device).repeat(means3D_final.shape[0], 1)
        # print("ratio:", ratio)
        gumbel_soft_mask, gumbel_hard_mask, _ = pc.gumbel_net(means3D_final, rotations_final, scales_final, time_gumbel)
        selected_mask_soft = gumbel_hard_mask
        selected_mask_hard = gumbel_hard_mask > 0
        
        print("Pred ratio:", torch.sum(selected_mask_hard) / torch.sum(torch.ones_like(selected_mask_hard)))
        print("Real ratio:", ratio)
        
        select_means3D = means3D_final[selected_mask_hard]
        # select_means2D = means2D[selected_mask]
        select_shs = shs_final[selected_mask_hard]
        select_opacity = opacity[selected_mask_hard]
        select_scales = scales_final[selected_mask_hard]
        select_rotations = rotations_final[selected_mask_hard]
        
        time = torch.tensor(ratio, dtype=torch.float32, requires_grad=True).to(select_means3D.device).repeat(select_means3D.shape[0], 1)
        
        # select_means3D, select_scales, select_rotations, _, _ = pc._deformation(
        #     select_means3D, select_scales, 
        #     select_rotations, select_opacity, select_shs,
        #     time
        # )
        
        
        means3D_gumbel = torch.zeros_like(means3D_final)
        scales_gumbel = torch.zeros_like(scales_final)
        rotations_gumbel = torch.zeros_like(rotations_final)

        means3D_gumbel[selected_mask_hard] = select_means3D
        scales_gumbel[selected_mask_hard] = select_scales
        rotations_gumbel[selected_mask_hard] = select_rotations
        
        means3D_gumbel[~selected_mask_hard] = means3D_final[~selected_mask_hard]
        scales_gumbel[~selected_mask_hard] = scales_final[~selected_mask_hard]
        rotations_gumbel[~selected_mask_hard] = rotations_final[~selected_mask_hard]
        
        opacity = opacity * selected_mask_soft.unsqueeze(1)
        
        selected_rendered_image, selected_radii = rasterizer(
            means3D = means3D_gumbel,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp,    # None
            opacities = opacity,
            scales = scales_gumbel,
            rotations = rotations_gumbel,
            cov3D_precomp = cov3D_precomp       # None
        )
    

    return {
        "render": selected_rendered_image,
        "visibility_filter" : selected_radii > 0,
        "radii": selected_radii,
        
        "full_render": full_rendered_image,
        "full_radii": full_radii,
        
        "gumbel_soft_mask": gumbel_soft_mask, 
        "gumbel_hard_mask": gumbel_hard_mask,
        
        "viewspace_points": screenspace_points,
    }


def select_render_for_save(
    viewpoint_camera, 
    pc : GaussianModel, 
    pipe, 
    bg_color : torch.Tensor, 
    scaling_modifier = 1.0, 
    override_color = None, 
    stage="fine", 
    cam_type=None,
    selected_mask=None, 
    ratio=1.0
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug,
            # f_count=True,
            f_count=False,
        )
        # time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
        # time = torch.tensor(ratio, dtype=torch.float32, requires_grad=True).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        # time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)
        # time = torch.tensor(ratio, requires_grad=True).to(means3D.device).repeat(means3D.shape[0],1)
        

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table

    # if stage == "coarse":
    #     means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations,opacity, shs
        
    # else:
    #     means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
    #         means3D, scales, 
    #         rotations, opacity, shs,
    #         time
    #     )

    means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations,opacity, shs
    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)

    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
    else:
        colors_precomp = override_color

    full_rendered_image, full_radii = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = colors_precomp,    # None
            opacities = opacity,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = cov3D_precomp       # None
        )
    
    select_means3D = means3D_final[selected_mask]
    select_means2D = means2D[selected_mask]
    select_shs = shs_final[selected_mask]
    select_opacity = opacity[selected_mask]
    select_scales = scales_final[selected_mask]
    select_rotations = rotations_final[selected_mask]
    
    selected_rendered_image, selected_radii = full_rendered_image, full_radii
    
    if ratio != 1: 
        time = torch.tensor(ratio, dtype=torch.float32, requires_grad=True).to(select_means3D.device).repeat(select_means3D.shape[0],1)
        # print("ratio:", ratio)
        select_means3D, select_scales, select_rotations, select_opacity, select_shs = pc._deformation(
            select_means3D, select_scales, 
            select_rotations, select_opacity, select_shs,
            time
        )
    
        selected_rendered_image, selected_radii = rasterizer(
            means3D = select_means3D,
            means2D = select_means2D,
            shs = select_shs,
            colors_precomp = colors_precomp,    # None
            opacities = select_opacity,
            scales = select_scales,
            rotations = select_rotations,
            cov3D_precomp = cov3D_precomp       # None
        )
    

    return {
        "render": selected_rendered_image,
        "visibility_filter" : selected_radii > 0,
        "radii": selected_radii,
        
        "full_render": full_rendered_image,
        "full_radii": full_radii,
        
        "viewspace_points": screenspace_points,
        # "gaussians_count": gaussians_count,
        
    }
    
