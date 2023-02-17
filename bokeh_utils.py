import os
import cv2
import time
import imageio
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

import nerf_utils
import nerf_model as nerf_model

from bokeh_renderer.scatter import ModuleRenderScatter
# from bokeh_renderer.scatter_ex import ModuleRenderScatterEX as ModuleRenderScatter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_blur(x, r, sigma=None):
    r = int(round(r))
    if sigma is None:
        sigma = 0.3 * (r - 1) + 0.8
    x_grid, y_grid = torch.meshgrid(torch.arange(-int(r), int(r) + 1), torch.arange(-int(r), int(r) + 1))
    kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / 2 / sigma ** 2)
    kernel = kernel.float() / kernel.sum()
    kernel = kernel.expand(1, 1, 2*r+1, 2*r+1).to(x.device)
    x = F.pad(x, pad=(r, r, r, r), mode='replicate')
    x = F.conv2d(x, weight=kernel, padding=0)
    return x


def render_bokeh(rgbs, 
                 disps, 
                 K_bokeh=20, 
                 gamma=4, 
                 disp_focus=90/255, 
                 defocus_scale=1):
    
    classical_renderer = ModuleRenderScatter().to(device)

    disps =  (disps - disps.min()) / (disps.max()- disps.min())
    # disps = disps / disps.max()
    
    signed_disp = disps - disp_focus
    defocus = (K_bokeh) * signed_disp / defocus_scale

    defocus = defocus.unsqueeze(0).unsqueeze(0).contiguous()
    rgbs = rgbs.permute(2, 0, 1).unsqueeze(0).contiguous()

    bokeh_classical = classical_renderer(rgbs**gamma, defocus*defocus_scale)
    bokeh_classical = bokeh_classical ** (1/gamma)
    bokeh_classical = bokeh_classical[0].permute(1, 2, 0)
    return bokeh_classical

def render_bokeh_wo_norm_disp(rgbs, 
                              disps,
                              K_bokeh=20, 
                              gamma=4, 
                              disp_focus=90/255, 
                              defocus_scale=1):
    
    classical_renderer = ModuleRenderScatter().to(device)

    # disps =  (disps - disps.min()) / (disps.ma x()- disps.min())
    # disps = disps / disps.max()
    
    signed_disp = disps - disp_focus
    defocus = (K_bokeh) * signed_disp / defocus_scale

    defocus = defocus.unsqueeze(0).unsqueeze(0).contiguous()
    rgbs = rgbs.permute(2, 0, 1).unsqueeze(0).contiguous()

    bokeh_classical = classical_renderer(rgbs**gamma, defocus*defocus_scale)
    bokeh_classical = bokeh_classical ** (1/gamma)
    bokeh_classical = bokeh_classical[0].permute(1, 2, 0)
    return bokeh_classical


def render_nerf_multi_bokeh(render_pose, 
                            hwf, 
                            K, 
                            chunk, 
                            render_kwargs, 
                            K_bokeh=3.6, 
                            gamma=4, 
                            disp_focus=30/255, 
                            change_focus=True,
                            defocus_scale=1,
                            gt_imgs=None, 
                            savedir=None, 
                            render_factor=0,
                            name=0):

    if change_focus:
        disp_focus = list(range(0,200,1))
    else:
        K_bokeh = list(range(0,30,3))

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    c2w = render_pose
    
    bokeh = []
    
    t = time.time()

    rgb, disp, acc, _ = nerf_utils.render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
    
    if change_focus:
    # for i,item in tqdm(enumerate(K_bokeh)):
        disp_focus = list(range(0,200,1))
        for i,item in tqdm(enumerate(disp_focus)):
            bokeh_classical = render_bokeh(rgb, 
                                           disp, 
                                           K_bokeh=K_bokeh,
                                           gamma=gamma,
                                           disp_focus=item/255,
                                           defocus_scale=defocus_scale)

            bokeh.append(bokeh_classical.cpu().numpy())

            if savedir is not None:
                bokeh8 = nerf_model.to8b(bokeh_classical.cpu().numpy())
                filename = os.path.join(savedir, '{}_pose_{}_param.png'.format(name, i))
                imageio.imwrite(filename, bokeh8)

        bokeh = np.stack(bokeh, 0)
    else:
        K_bokeh = list(range(0,30,3))
        for i,item in tqdm(enumerate(K_bokeh)):
            bokeh_classical = render_bokeh(rgb, 
                                           disp, 
                                           K_bokeh=item,
                                           gamma=gamma,
                                           disp_focus=disp_focus,
                                           defocus_scale=defocus_scale)

            bokeh.append(bokeh_classical.cpu().numpy())

            if savedir is not None:
                bokeh8 = nerf_model.to8b(bokeh_classical.cpu().numpy())
                filename = os.path.join(savedir, '{}_pose_{}_param.png'.format(name, i))
                imageio.imwrite(filename, bokeh8)

        bokeh = np.stack(bokeh, 0)

    return bokeh


def render_path_bokeh(render_poses, 
                      hwf, 
                      K, 
                      chunk, 
                      render_kwargs,
                      K_bokeh=1., 
                      gamma=4, 
                      disp_focus=30/255, 
                      defocus_scale=1, 
                      gt_imgs=None, 
                      savedir=None, 
                      render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb_0, disp, acc, _ = nerf_utils.render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        
        rgb = render_bokeh(rgb_0, 
                           disp, 
                           K_bokeh=K_bokeh,
                           gamma=gamma,
                           disp_focus=disp_focus,
                           defocus_scale=defocus_scale)
        
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)


        if savedir is not None:
            rgb8 = nerf_model.to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def render_path_bokeh_mod(render_poses, 
                          hwf, 
                          K, 
                          chunk, 
                          render_kwargs,
                          K_bokeh=1., 
                          gamma=4, 
                          disp_focus=30/255, 
                          defocus_scale=1, 
                          gt_imgs=None, 
                          savedir=None, 
                          render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs_0 = None
    disps_0 = None
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        # print(i, time.time() - t)
        t = time.time()
        rgb_0, disp_0, acc, _ = nerf_utils.render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        
        # rgbs_0.append(rgb_0.cpu().numpy())
        disps.append(disp_0.cpu().numpy())
        # if not rgbs_0:
        #     rgbs_0 = rgb_0.unsqueeze(0)
        #     disps_0 = disp_0.unsqueeze(0)
        # else:
        #     rgbs_0 = torch.cat((rgbs_0, rgb_0.unsqueeze(0)), 0)
        #     disps_0 = torch.cat((disps_0, disp_0.unsqueeze(0)), 0)
        if rgbs_0 is not None:
            rgbs_0 = torch.cat((rgbs_0, rgb_0.unsqueeze(0)), 0)
            disps_0 = torch.cat((disps_0, disp_0.unsqueeze(0)), 0)
        else:
            rgbs_0 = rgb_0.unsqueeze(0)
            disps_0 = disp_0.unsqueeze(0)
        
        if i==0:
            print(rgb_0.shape, disp_0.shape)

        if savedir is not None:
            rgb8 = nerf_model.to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    max_disp = disps_0.max()
    print('max_disp:{}'.format(max_disp))

    rgbs = []
    for i in range(rgbs_0.shape[0]):

        rgb = render_bokeh_wo_norm_disp(rgbs_0[i, ...], 
                                        disps_0[i, ...] / max_disp, 
                                        K_bokeh=K_bokeh,
                                        gamma=gamma,
                                        disp_focus=disp_focus,
                                        defocus_scale=defocus_scale)

        rgbs.append(rgb.cpu().numpy())

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps