from itertools import chain
import matplotlib.pyplot as plt
from types import SimpleNamespace
from nvdiffrast.torch import RasterizeGLContext, rasterize

from model.utils.misc import *
from model.render.renderutils import xfm_points
from model.models.Fauna import FaunaConfig, FaunaModel
from model.predictors.InstancePredictorBase import InstancePredictorBase
from model.predictors.BasePredictorFauna4D import BasePredictorFauna4D
from model.predictors.InstancePredictorFauna4D import InstancePredictorFauna4D


class FaunaFinetune(FaunaModel):
    def __init__(self, cfg: FaunaConfig):
        super().__init__(cfg)

    def set_train(self):
        super().set_train()
        for param in chain(self.netInstance.parameters(), self.netBase.parameters(), self.netDisc.parameters()):
            param.requires_grad = False
        for param in chain(
            self.netInstance.netArticulation.parameters(),
            # self.netDisc.parameters()
        ):
            param.requires_grad = True
        # if self.netInstance.enable_deform:
        #     for param in self.netInstance.netDeform.parameters():
        #         param.requires_grad = True

    def set_eval(self):
        super().set_eval()

    def compute_regularizers(
        self, arti_params=None, deformation=None, pose_raw=None, posed_bones=None, class_vector=None, prior_shape=None,
        instance_shape=None, keypoint_gt=None, keypoint_gt_flag=None, mvp=None, **kwargs
    ):
        losses, aux = super().compute_regularizers(arti_params, deformation, pose_raw, posed_bones, class_vector, prior_shape)
        # Prior surface normal regularization
        if prior_shape is not None:
            adj_verts = torch.cat([prior_shape.t_nrm_idx[:,:,0:2], prior_shape.t_nrm_idx[:,:,1:3]], dim=1)  # (1, 2*n_faces, 2)
            adj_vert_x_coords = torch.mean(prior_shape.v_pos[:, adj_verts].squeeze(1), dim=-2)[:, :,1]  # (1, 2*n_faces)
            adj_norms = prior_shape.v_nrm[:, adj_verts].squeeze(1)  # (1, 2*n_faces, 2, 3)
            adj_norm_diffs = 1 - torch.sum(adj_norms[:,:,0,:] * adj_norms[:,:,1,:], dim=-1)
            xmin = adj_vert_x_coords.min(dim=-1).values
            xmax = adj_vert_x_coords.max(dim=-1).values
            xmid = (xmin + xmax) / 2
            radius = (xmax - xmin) / 2
            weights = torch.sqrt(torch.clamp(radius**2 - (xmid - adj_vert_x_coords)**2, min=1e-8)) / radius
            weights = torch.ones_like(weights)
            losses['prior_normal_reg_loss'] = torch.sum(adj_norm_diffs * weights) / torch.sum(weights)
        # Instance shape normal regularization, only works when all shapes in batch are from same prior shape
        if instance_shape is not None:
            adj_verts = torch.cat([instance_shape.t_nrm_idx[:,:,0:2], instance_shape.t_nrm_idx[:,:,1:3]], dim=1)  # (1, 2*n_faces, 2)
            adj_vert_x_coords = torch.mean(instance_shape.v_pos[:, adj_verts].squeeze(1), dim=-2)[:, :,1]  # (b, 2*n_faces)
            adj_norms = instance_shape.v_nrm[:, adj_verts].squeeze(1)  # (b, 2*n_faces, 2, 3)
            adj_norm_diffs = 1 - torch.sum(adj_norms[:,:,0,:] * adj_norms[:,:,1,:], dim=-1)
            xmin = adj_vert_x_coords.min(dim=-1).values
            xmax = adj_vert_x_coords.max(dim=-1).values
            xmid = (xmin + xmax)[:, None] / 2
            radius = (xmax - xmin)[:, None] / 2
            weights = torch.sqrt(torch.clamp(radius**2 - (xmid - adj_vert_x_coords)**2, min=1e-8)) / radius
            weights = torch.ones_like(weights)
            losses['instance_normal_reg_loss'] = torch.sum(adj_norm_diffs * weights) / torch.sum(weights)
        # Keypoint projection loss
        if keypoint_gt is not None and posed_bones is not None and keypoint_gt_flag.any():
            assert mvp is not None
            # pred_to_gt_map = {8: 7, 9: 6, 10: 5, 11: 13, 12: 12, 13: 11, 14: 16, 15: 15, 16: 14, 17: 10, 18: 9, 19: 8, 0: 2}
            # Only feet, keen and nose keypoints
            pred_to_gt_map = {8: 7, 9: 6, 11: 13, 12: 12, 14: 16, 15: 15, 17: 10, 18: 9, 0: 2}
            bone_world4 = torch.concat(
                [posed_bones, torch.ones_like(posed_bones[..., :1]).to(posed_bones.device)], dim=-1
            )
            b, f, num_bones = bone_world4.shape[:3]
            bones_clip4 = (
                bone_world4.view(b, f, num_bones * 2, 1, 4) @ mvp.transpose(-1, -2).reshape(b, f, 1, 4, 4)
            ).view(b, f, num_bones, 2, 4)
            bones_uvd = bones_clip4[..., :3] / bones_clip4[..., 3:4]  # b, f, num_bones, 2, 3
            keypoint_pred = bones_uvd[:, :, list(pred_to_gt_map.keys()), -1, :2]  # b, f, k, 2
            keypoint_gt = keypoint_gt[:, :, list(pred_to_gt_map.values()), :2]  # b, f, k, 2
            losses["keypoint_projection_loss"] = ((keypoint_pred - keypoint_gt)[keypoint_gt_flag] ** 2).mean()
        return losses, aux

    def forward(self, batch, epoch, logger=None, total_iter=None, save_results=False, save_dir=None, logger_prefix='',
                is_training=True):
        input_image, mask_gt, mask_dt, mask_valid, flow_gt, bbox, bg_image, dino_feat_im, dino_cluster_im, keypoint_gt, seq_idx, frame_idx = batch
        if bbox.shape[2] == 9:
            # Fauna Dataset bbox
            global_frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness, tmp_label = bbox.unbind(
                2)  # BxFx9
        elif bbox.shape[2] == 8:
            # in visualization using magicpony dataset for simplicity
            global_frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness = bbox.unbind(2)  # BxFx8
        else:
            raise NotImplementedError

        mask_gt = (mask_gt[:, :, 0, :, :] > 0.9).float()  # BxFxHxW
        mask_dt = mask_dt / self.dataset.in_image_size
        batch_size, num_frames, _, _, _ = input_image.shape  # BxFxCxHxW
        h = w = self.dataset.out_image_size
        aux_viz = {}

        dino_feat_im_gt = None if dino_feat_im is None else expandBF(
            torch.nn.functional.interpolate(collapseBF(dino_feat_im), size=[h, w], mode="bilinear"), batch_size,
            num_frames)[:, :, :self.cfg_predictor_base.cfg_dino.feature_dim]
        dino_cluster_im_gt = None if dino_cluster_im is None else expandBF(
            torch.nn.functional.interpolate(collapseBF(dino_cluster_im), size=[h, w], mode="nearest"), batch_size,
            num_frames)

        ## GT image
        image_gt = input_image
        if self.dataset.out_image_size != self.dataset.in_image_size:
            image_gt = expandBF(torch.nn.functional.interpolate(collapseBF(image_gt), size=[h, w], mode='bilinear'),
                                batch_size, num_frames)
            if flow_gt is not None:
                flow_gt = expandBF(torch.nn.functional.interpolate(collapseBF(flow_gt), size=[h, w], mode="bilinear"),
                                   batch_size, num_frames - 1)

        ## predict prior shape and DINO
        if in_range(total_iter, self.cfg_predictor_base.cfg_shape.grid_res_coarse_iter_range):
            grid_res = self.cfg_predictor_base.cfg_shape.grid_res_coarse
        else:
            grid_res = self.cfg_predictor_base.cfg_shape.grid_res
        if self.get_predictor("netBase").netShape.grid_res != grid_res:
            self.get_predictor("netBase").netShape.load_tets(grid_res)
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                prior_shape, dino_net, bank_embedding = self.netBase(total_iter=total_iter, is_training=is_training,
                                                                     batch=batch, bank_enc=self.get_predictor(
                        "netInstance").netEncoder)
        else:
            prior_shape, dino_net, bank_embedding = self.netBase(total_iter=total_iter, is_training=is_training,
                                                                 batch=batch,
                                                                 bank_enc=self.get_predictor("netInstance").netEncoder)

        class_vector = bank_embedding[0]

        ## predict instance specific parameters
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(
                    input_image, prior_shape, epoch, total_iter, frame_ids=frame_idx, is_training=is_training)
            pose_raw, pose, mvp, w2c, campos, im_features, arti_params = \
                map(to_float, [pose_raw, pose, mvp, w2c, campos, im_features, arti_params])
        else:
            shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(
                input_image, prior_shape, epoch, total_iter, frame_ids=frame_idx, is_training=is_training)
        keypoint_gt_flag = disable_keypoint_loss(w2c, b=batch_size)
        print(keypoint_gt_flag)
        keypoint_gt_flag = rearrange(keypoint_gt_flag, "(b f) -> b f", b=batch_size, f=num_frames)
        rot_logit = forward_aux['rot_logit']
        rot_idx = forward_aux['rot_idx']
        rot_prob = forward_aux['rot_prob']
        aux_viz.update(forward_aux)
        final_losses = {}

        ## render images
        if self.enable_render or not is_training:  # Force render for val and test
            render_flow = self.cfg_render.render_flow and num_frames > 1
            render_modes = ['shaded', 'dino_pred']
            if render_flow:
                render_modes += ['flow']
            if self.mixed_precision:
                with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                    renders = self.render(
                        render_modes, shape, texture, mvp, w2c, campos, (h, w), im_features=im_features, light=light,
                        prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames,
                        class_vector=class_vector[None, :].expand(batch_size * num_frames, -1)
                    )
            else:
                renders = self.render(
                    render_modes, shape, texture, mvp, w2c, campos, (h, w), im_features=im_features, light=light,
                    prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames,
                    class_vector=class_vector[None, :].expand(batch_size * num_frames, -1)
                )
            renders = map(lambda x: expandBF(x, batch_size, num_frames), renders)
            if render_flow:
                shaded, dino_feat_im_pred, flow_pred = renders
                flow_pred = flow_pred[:, :-1]  # Bx(F-1)x2xHxW
            else:
                shaded, dino_feat_im_pred = renders
                flow_pred = None
            image_pred = shaded[:, :, :3]
            mask_pred = shaded[:, :, 3]

            ## compute reconstruction losses
            if self.mixed_precision:
                with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                    losses = self.compute_reconstruction_losses(
                        image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt,
                        dino_feat_im_gt,
                        dino_feat_im_pred, background_mode=self.cfg_render.background_mode, reduce=False
                    )
            else:
                losses = self.compute_reconstruction_losses(
                    image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt, dino_feat_im_gt,
                    dino_feat_im_pred, background_mode=self.cfg_render.background_mode, reduce=False
                )

            ## supervise the rotation logits directly with reconstruction loss
            logit_loss_target = None
            if losses is not None:
                logit_loss_target = torch.zeros_like(expandBF(rot_logit, batch_size, num_frames))
                for name, loss in losses.items():
                    loss_weight = getattr(self.cfg_loss, f"{name}_weight")
                    if name in ['dino_feat_im_loss']:
                        ## increase the importance of dino loss for viewpoint hypothesis selection (directly increasing dino recon loss leads to stripe artifacts)
                        # loss_weight = loss_weight * self.cfg_loss.logit_loss_dino_feat_im_loss_multiplier
                        loss_weight = self.parse_dict_definition(self.cfg_loss.dino_feat_im_loss_weight_dict,
                                                                 total_iter)
                        loss_weight = loss_weight * self.parse_dict_definition(
                            self.cfg_loss.logit_loss_dino_feat_im_loss_multiplier_dict, total_iter)
                    if name in ['mask_loss']:
                        loss_weight = loss_weight * self.cfg_loss.logit_loss_mask_multiplier
                    if name in ['mask_inv_dt_loss']:
                        loss_weight = loss_weight * self.cfg_loss.logit_loss_mask_inv_dt_multiplier
                    if loss_weight > 0:
                        logit_loss_target += loss * loss_weight

                    ## multiply the loss with probability of the rotation hypothesis (detached)
                    if self.get_predictor("netInstance").cfg_pose.rot_rep in ['quadlookat', 'octlookat']:
                        loss_prob = rot_prob.detach().view(batch_size, num_frames)[:,
                                    :loss.shape[1]]  # handle edge case for flow loss with one frame less
                        loss = loss * loss_prob * self.get_predictor("netInstance").num_pose_hypos
                    ## only compute flow loss for frames with the same rotation hypothesis
                    if name == 'flow_loss' and num_frames > 1:
                        ri = rot_idx.view(batch_size, num_frames)
                        same_rot_idx = (ri[:, 1:] == ri[:, :-1]).float()
                        loss = loss * same_rot_idx
                    ## update the final prob-adjusted losses
                    final_losses[name] = loss.mean()

                logit_loss_target = collapseBF(logit_loss_target).detach()  # detach the gradient for the loss target
                final_losses['logit_loss'] = ((rot_logit - logit_loss_target) ** 2.).mean()
                final_losses['logit_loss_target'] = logit_loss_target.mean()

        random_view_aux = None
        random_view_aux = self.get_random_view_mask(w2c, shape, prior_shape, num_frames)
        if (self.cfg_mask_discriminator.enable_iter[0] < total_iter) and (
                self.cfg_mask_discriminator.enable_iter[1] > total_iter):
            disc_loss = self.compute_mask_disc_loss_gen(
                mask_gt, mask_pred, random_view_aux['mask_random_pred'], condition_feat=class_vector
            )
            final_losses.update(disc_loss)

        ## regularizers
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                regularizers, aux = self.compute_regularizers(
                    arti_params=arti_params, deformation=deformation, pose_raw=pose_raw, mvp=mvp, instance_shape=shape,
                    prior_shape=prior_shape, posed_bones=forward_aux.get("posed_bones"), keypoint_gt=keypoint_gt,
                    keypoint_gt_flag=keypoint_gt_flag,
                    class_vector=class_vector.detach() if class_vector is not None else None
                )
        else:
            regularizers, aux = self.compute_regularizers(
                arti_params=arti_params, deformation=deformation, pose_raw=pose_raw, mvp=mvp, instance_shape=shape,
                prior_shape=prior_shape, posed_bones=forward_aux.get("posed_bones"), keypoint_gt=keypoint_gt,
                keypoint_gt_flag=keypoint_gt_flag,
                class_vector=class_vector.detach() if class_vector is not None else None
            )
        final_losses.update(regularizers)
        aux_viz.update(aux)

        ## compute final losses
        total_loss = 0
        for name, loss in final_losses.items():
            loss_weight = getattr(self.cfg_loss, f"{name}_weight")
            if loss_weight <= 0:
                continue
            if not in_range(total_iter, self.cfg_predictor_instance.cfg_texture.texture_iter_range) and (
                    name in ['rgb_loss']):
                continue
            if not in_range(total_iter, self.cfg_loss.arti_reg_loss_iter_range) and (name in ['arti_reg_loss']):
                continue
            if name in ["logit_loss_target"]:
                continue

            if name == 'dino_feat_im_loss':
                loss_weight = self.parse_dict_definition(self.cfg_loss.dino_feat_im_loss_weight_dict, total_iter)

            total_loss += loss * loss_weight
        self.total_loss += total_loss  # reset to 0 in backward step

        if torch.isnan(self.total_loss):
            print("NaN in loss...")
            import pdb;
            pdb.set_trace()

        metrics = {'loss': total_loss, **final_losses}

        log = SimpleNamespace(**locals())
        if logger is not None and (self.enable_render or not is_training):
            self.log_visuals(log, logger)
        if save_results:
            self.save_results(log)
        return metrics

    @torch.no_grad()
    def log_visuals(self, log, logger):
        # Add keypoint visualization to image_gt
        if log.keypoint_gt is not None:
            log.input_image = draw_keypoints(log.input_image, log.keypoint_gt, log.keypoint_gt_flag)
        # return super().log_visuals(log, logger)  # Will OOM

        text = None
        b0 = max(min(log.batch_size, 16 // log.num_frames), 1)
        def log_image(name, image):
            logger.add_image(log.logger_prefix + 'image/' + name,
                             image_grid(collapseBF(image[:b0, :]).detach().cpu().clamp(0, 1)), log.total_iter)
        def log_video(name, frames, fps=2):
            logger.add_video(log.logger_prefix + 'animation/' + name, frames.detach().cpu().unsqueeze(0).clamp(0, 1),
                             log.total_iter, fps=fps)
        log_image('image_gt', log.input_image)
        log_image('image_pred', log.image_pred)
        log_image('mask_gt', log.mask_gt.unsqueeze(2).repeat(1, 1, 3, 1, 1))
        log_image('mask_pred', log.mask_pred.unsqueeze(2).repeat(1, 1, 3, 1, 1))


        if log.dino_feat_im_gt is not None:
            log_image('dino_feat_im_gt', log.dino_feat_im_gt[:, :, :3])
        if log.dino_feat_im_pred is not None:
            log_image('dino_feat_im_pred', log.dino_feat_im_pred[:, :, :3])
        if log.dino_cluster_im_gt is not None:
            log_image('dino_cluster_im_gt', log.dino_cluster_im_gt)

        render_modes = ['geo_normal', 'kd', 'shading']
        rendered = self.render(render_modes, log.shape, log.texture, log.mvp, log.w2c, log.campos, (log.h, log.w),
                               im_features=log.im_features, light=log.light, prior_shape=log.prior_shape)
        geo_normal, albedo, shading = map(lambda x: expandBF(x, log.batch_size, log.num_frames), rendered)
        if hasattr(self.get_predictor("netInstance"), "articulated_shape_gt"):
            rendered_gt = self.render(render_modes, self.get_predictor("netInstance").articulated_shape_gt, log.texture,
                                      log.mvp, log.w2c, log.campos, (log.h, log.w), im_features=log.im_features,
                                      light=log.light, prior_shape=log.prior_shape)
            geo_normal_gt, albedo_gt, shading_gt = map(lambda x: expandBF(x, log.batch_size, log.num_frames),
                                                       rendered_gt)
            del self.get_predictor("netInstance").articulated_shape_gt

        if log.light is not None:
            param_names = ['dir_x', 'dir_y', 'dir_z', 'int_ambient', 'int_diffuse']
            for name, param in zip(param_names, log.light.light_params.unbind(-1)):
                logger.add_histogram(log.logger_prefix + 'light/' + name, param, log.total_iter)
            log_image('albedo', albedo)
            log_image('shading', shading.repeat(1, 1, 3, 1, 1) / 2.)

        ## add bone visualizations
        if 'posed_bones' in log.aux_viz:
            rendered_bone_image = self.render_bones(log.mvp, log.aux_viz['posed_bones'], (log.h, log.w))
            rendered_bone_image_mask = (rendered_bone_image < 1).float()
            geo_normal = rendered_bone_image_mask * 0.8 * rendered_bone_image + (
                        1 - rendered_bone_image_mask * 0.8) * geo_normal
            if log.aux_viz.get("posed_bones_gt") is not None:
                rendered_bone_image = self.render_bones(log.mvp, log.aux_viz['posed_bones_gt'], (log.h, log.w))
                rendered_bone_image_mask = (rendered_bone_image < 1).float()
                geo_normal_gt = rendered_bone_image_mask * 0.8 * rendered_bone_image + (
                        1 - rendered_bone_image_mask * 0.8) * geo_normal_gt
                log_image('instance_geo_normal_gt', geo_normal_gt)

        ## draw marker on images with randomly sampled pose
        if self.cfg_predictor_instance.cfg_pose.rot_rep in ['quadlookat', 'octlookat']:
            rand_pose_flag = log.forward_aux['rand_pose_flag']
            rand_pose_marker_mask = torch.zeros_like(geo_normal)
            rand_pose_marker_mask[:, :, :, :16, :16] = 1.
            rand_pose_marker_mask = rand_pose_marker_mask * rand_pose_flag.view(log.batch_size, log.num_frames, 1, 1, 1)
            red = torch.FloatTensor([1, 0, 0]).view(1, 1, 3, 1, 1).to(geo_normal.device)
            geo_normal = rand_pose_marker_mask * red + (1 - rand_pose_marker_mask) * geo_normal

        log_image('instance_geo_normal', geo_normal)
        rot_frames = self.render_rotation_frames('geo_normal', log.shape, log.texture, log.light, (log.h, log.w),
                                                 im_features=log.im_features, prior_shape=log.prior_shape,
                                                 num_frames=15, b=1, text=text)
        log_video('instance_normal_rotation', rot_frames)
        rot_frames = self.render_rotation_frames('shaded', log.prior_shape, log.texture, log.light, (log.h, log.w),
                                                 im_features=log.im_features, num_frames=15, b=1, text=text)
        log_video('prior_image_rotation', rot_frames)
        rot_frames = self.render_rotation_frames('geo_normal', log.prior_shape, log.texture, log.light, (log.h, log.w),
                                                 im_features=log.im_features, num_frames=15, b=1, text=text)
        log_video('prior_normal_rotation', rot_frames)
        log.__dict__.update({k: v for k, v in locals().items() if k != "log"})
        return log




class Fauna4DModel(FaunaFinetune):
    def __init__(self, cfg: FaunaConfig):
        super().__init__(cfg)
        self.netBase = BasePredictorFauna4D(self.cfg_predictor_base)
        self.netInstance = InstancePredictorFauna4D(self.cfg_predictor_instance)

    @torch.no_grad()
    def compute_mean_feature(self, dataloader):
        netBase = self.get_predictor("netBase")
        encoder = self.get_predictor("netInstance").netEncoder
        if netBase.mean_feature is not None:
            return
        num_features = 0
        mean_feature = None
        for batch in dataloader:
            images = batch[0]
            b, f, c, h, w = images.shape
            images = rearrange(images, "b f ... -> (b f) ...")
            images_in = images * 2 - 1
            batch_features = netBase.forward_frozen_ViT(images_in, encoder)
            batch_embedding, _, _ = netBase.retrieve_memory_bank(batch_features, batch)
            if mean_feature is None:
                mean_feature = batch_embedding
            else:
                mean_feature += batch_embedding
            num_features += b * f
        mean_feature = mean_feature / num_features
        netBase.mean_feature = mean_feature
        return mean_feature

    def set_finetune_arti(self):
        super().set_train()
        for param in chain(self.netInstance.parameters(), self.netBase.parameters(), self.netDisc.parameters()):
            param.requires_grad = False
        for param in chain(
            self.netInstance.articulation_dict.parameters(),
            self.netInstance.pose_dict.parameters(),
            # self.netBase.parameters(),
        ):
            param.requires_grad = True
        if self.netInstance.enable_deform:
            for param in self.netInstance.netDeform.parameters():
                param.requires_grad = True
        # Set a learnable fov
        self.netInstance.fov = torch.nn.Parameter(torch.tensor([self.netInstance.cfg_pose.fov], dtype=torch.float32))
        self.netInstance.get_camera_extrinsics_from_pose = get_camera_extrinsics_from_pose_differentiable.__get__(self.netInstance, InstancePredictorBase)
        self.netInstance.enable_deform = False
        self.optimizerInstance.add_param_group({"params": [self.netInstance.fov]})

    def set_finetune_texture(self):
        super().set_train()
        for param in chain(self.netInstance.parameters(), self.netBase.parameters(), self.netDisc.parameters()):
            param.requires_grad = False
        for param in chain(self.netInstance.netTexture.parameters()):
            param.requires_grad = True
        self.netInstance.enable_deform = True

    def set_inference(self):
        super().set_eval()
        for param in chain(self.netInstance.parameters(), self.netBase.parameters()):
            param.requires_grad = False
        assert self.get_predictor("netBase").mean_feature is not None


    # def get_default_pose(self):
    #     pose_canon = torch.concat([torch.eye(3), torch.zeros(1, 3)], dim=0).view(-1)[None].to(self.device)
    #     mvp_canon, w2c_canon, campos_canon = self.netInstance.get_camera_extrinsics_from_pose(pose_canon, offset_extra=self.cfg_render.offset_extra)
    #     viewpoint_arti = torch.FloatTensor([0, -120, 0]) / 180 * np.pi
    #     mtx = torch.eye(4).to(self.device)
    #     mtx[:3, :3] = euler_angles_to_matrix(viewpoint_arti, "XYZ")
    #     w2c_arti = torch.matmul(w2c_canon, mtx[None])
    #     mvp_arti = torch.matmul(mvp_canon, mtx[None])
    #     campos_arti = campos_canon @ torch.linalg.inv(mtx[:3, :3]).T
    #     self.default_pose, self.default_mvp, self.default_w2c, self.default_campos = pose_canon, mvp_arti, w2c_arti, campos_arti


    @torch.no_grad()
    def log_visuals(self, log, logger):
        log = super().log_visuals(log, logger)
        b0 = max(min(log.batch_size, 16 // log.num_frames), 1)
        def log_image(name, image):
            logger.add_image(log.logger_prefix + 'image/' + name, image_grid(collapseBF(image[:b0, :]).detach().cpu().clamp(0, 1)), log.total_iter)
        def log_video(name, frames, fps=5):
            logger.add_video(log.logger_prefix+'animation/'+name, frames.detach().cpu().unsqueeze(0).clamp(0,1), log.total_iter, fps=fps)
        if log.num_frames > 1:
            log_video("sequence_image_gt", log.input_image[0])
            log_video("sequence_mask_gt", repeat(log.mask_gt[0], "f h w -> f c h w", c=3))
            suffix = "pred"
            log_video(f"sequence_image_{suffix}", log.image_pred[0])
            log_video(f"sequence_mask_{suffix}", repeat(log.mask_pred[0], "f h w -> f c h w", c=3))
            log_video(f"sequence_instance_geo_normal_{suffix}", log.geo_normal[0])
            if hasattr(log, "geo_normal_gt"):
                log_video(f"sequence_instance_geo_normal_gt", log.geo_normal_gt[0])

            leg_bones = [8, 11, 14, 17]
            all_leg_bone_pos = []
            for frame_ids, posed_bones in zip(log.global_frame_id, log.aux_viz['posed_bones']):  # Iterate batch
                frame_id_to_posed_bones = {}
                for frame_id, posed_bones in zip(frame_ids, posed_bones):  # Get unique frames
                    frame_id_to_posed_bones[int(frame_id.item())] = posed_bones  # (20,2,3)
                fig, ax = plt.subplots(figsize=(5, 5))
                frame_ids = sorted(frame_id_to_posed_bones.keys())
                for bone_idx in leg_bones:
                    motion = [frame_id_to_posed_bones[frame_id][bone_idx, 0, 2].item() for frame_id in frame_ids]
                    ax.plot(frame_ids, motion, 'o-', label=bone_idx)
                ax.legend()
                fig.canvas.draw()
                img_str = fig.canvas.tostring_rgb()
                width, height = fig.canvas.get_width_height()
                plt.close(fig)
                img = np.frombuffer(img_str, dtype=np.uint8).reshape(height, width, 3) / 255.
                all_leg_bone_pos.append(rearrange(img, "h w c -> 1 c h w"))
            all_leg_bone_pos = torch.FloatTensor(np.stack(all_leg_bone_pos))
            log_image(f"leg_bone_pos", all_leg_bone_pos)

    def forward_finetune_arti(self, batch, epoch, logger=None, total_iter=None, save_results=False, save_dir=None, logger_prefix='', is_training=True, **kwargs):
        batch = batch[:-1]  # exclude path
        m = super().forward(batch, epoch, logger, total_iter, save_results, save_dir, logger_prefix, is_training)
        # Add newly added per frame params to optimizer
        for name in ["articulation", "pose"]:
            existing_frames = [
                d.get("frame_idx") for d in self.optimizerInstance.param_groups
                if d.get("frame_idx") is not None and d.get("name") == name]
            param_dict = getattr(self.netInstance, f"{name}_dict")
            for frame_idx, param in param_dict.items():
                if int(frame_idx) not in existing_frames:
                    self.optimizerInstance.add_param_group(
                        {"params": [param], "name": name, "frame_idx": int(frame_idx)}
                    )
        print("fov", self.netInstance.fov.item())
        return m

    def forward_finetune_texture(self, batch, epoch, logger=None, total_iter=None, save_results=False, save_dir=None, logger_prefix='', is_training=True, **kwargs):
        batch = batch[:-1]
        m = super().forward(batch, epoch, logger, total_iter, save_results, save_dir, logger_prefix, is_training)
        return m

    @torch.no_grad()
    def inference(self, batch, total_iter, epoch, local_save_dir=None, is_training=False, image_suffix=None, **kwargs):
        input_image, mask_gt, mask_dt, mask_valid, flow_gt, bbox, bg_image, dino_feat_im, dino_cluster_im, keypoint_gt, seq_idx, frame_idx, paths = batch
        if bbox.shape[2] == 9:
            # Fauna Dataset bbox
            global_frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness, tmp_label = bbox.unbind(
                2)  # BxFx9
        elif bbox.shape[2] == 8:
            # in visualization using magicpony dataset for simplicity
            global_frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness = bbox.unbind(2)  # BxFx8
        else:
            raise NotImplementedError

        mask_gt = (mask_gt[:, :, 0, :, :] > 0.9).float()  # BxFxHxW
        mask_dt = mask_dt / self.dataset.in_image_size
        batch_size, num_frames, _, _, _ = input_image.shape  # BxFxCxHxW
        h = w = self.dataset.out_image_size
        aux_viz = {}

        ## GT image
        image_gt = input_image
        if self.dataset.out_image_size != self.dataset.in_image_size:
            image_gt = expandBF(torch.nn.functional.interpolate(collapseBF(image_gt), size=[h, w], mode='bilinear'),
                                batch_size, num_frames)
            if flow_gt is not None:
                flow_gt = expandBF(torch.nn.functional.interpolate(collapseBF(flow_gt), size=[h, w], mode="bilinear"),
                                   batch_size, num_frames - 1)

        ## predict prior shape and DINO
        if in_range(total_iter, self.cfg_predictor_base.cfg_shape.grid_res_coarse_iter_range):
            grid_res = self.cfg_predictor_base.cfg_shape.grid_res_coarse
        else:
            grid_res = self.cfg_predictor_base.cfg_shape.grid_res
        if self.get_predictor("netBase").netShape.grid_res != grid_res:
            self.get_predictor("netBase").netShape.load_tets(grid_res)
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                prior_shape, dino_net, bank_embedding = self.netBase(total_iter=total_iter, is_training=is_training,
                                                                     batch=batch, bank_enc=self.get_predictor(
                        "netInstance").netEncoder)
        else:
            prior_shape, dino_net, bank_embedding = self.netBase(total_iter=total_iter, is_training=is_training,
                                                                 batch=batch,
                                                                 bank_enc=self.get_predictor("netInstance").netEncoder)

        class_vector = bank_embedding[0]

        ## predict instance specific parameters
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(
                    input_image, prior_shape, epoch, total_iter, frame_ids=frame_idx, is_training=is_training)
            pose_raw, pose, mvp, w2c, campos, im_features, arti_params = \
                map(to_float, [pose_raw, pose, mvp, w2c, campos, im_features, arti_params])
        else:
            shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(
                input_image, prior_shape, epoch, total_iter, frame_ids=frame_idx, is_training=is_training)
        # if not is_training and (batch_size != arti_params.shape[0] or num_frames != arti_params.shape[1]):
        #     # If b f sampled from vae different from training b f
        #     batch_size, num_frames = arti_params.shape[:2]
        rot_logit = forward_aux['rot_logit']
        rot_idx = forward_aux['rot_idx']
        rot_prob = forward_aux['rot_prob']
        aux_viz.update(forward_aux)
        final_losses = {}

        ## render images
        from visualization.visualize_results_fauna import FixedDirectionLight
        light = FixedDirectionLight(
            direction=torch.FloatTensor([0, 0, 1]).to(self.accelerator.device), amb=0.2, diff=0.7
        )
        render_flow = self.cfg_render.render_flow and num_frames > 1
        render_modes = ['geo_normal', 'shaded', 'dino_pred', 'shading']
        if render_flow:
            render_modes += ['flow']
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                renders = self.render(
                    render_modes, shape, texture, mvp, w2c, campos, (h, w), im_features=im_features, light=light,
                    prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames, background=None,
                    class_vector=class_vector[None, :].expand(batch_size * num_frames, -1)
                )
        else:
            renders = self.render(
                render_modes, shape, texture, mvp, w2c, campos, (h, w), im_features=im_features, light=light,
                prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames, background=None,
                class_vector=class_vector[None, :].expand(batch_size * num_frames, -1)
            )
        b0 = batch_size * num_frames
        if b0 != renders[0].shape[0]:
            batch_size = int(renders[0].shape[0] / num_frames)
        renders = map(lambda x: expandBF(x, batch_size, num_frames), renders)
        geo_normal, shaded, dino_feat_im_pred, shading = renders
        image_pred = shaded[:, :, :3]
        mask_pred = shaded[:, :, 3]
        mask_diff_img = mask_diff(mask_pred, mask_gt)
        bones = forward_aux["posed_bones"]
        rendered_bones = self.render_bones(mvp, bones, (image_pred.shape[-2], image_pred.shape[-1]))
        rendered_bone_image_mask = (rendered_bones < 1).any(dim=-3, keepdim=True).float()
        shading_with_bones = repeat(shading, "b f 1 h w -> b f 3 h w").clone()
        shading_with_bones = rendered_bone_image_mask * 0.8 * rendered_bones + (1 - rendered_bone_image_mask * 0.8) * shading_with_bones
        shading = remove_background(shading)
        shading_with_bones = remove_background(shading_with_bones)
        geo_normal = remove_background(geo_normal)
        rgb_overlayed = image_gt * (1 - mask_pred.unsqueeze(-3)).abs() + shading_with_bones[:,:,:3,...] * mask_pred.unsqueeze(-3)
        paths = [item for sublist in paths for item in sublist]
        keypoint_3d = get_keypoint_coordinates(bones, mvp)
        eval_aux = get_eval_aux(shape, mvp, resolution=(h, w))  # (bf, v, 4)
        if not os.path.isdir(local_save_dir):
            os.makedirs(local_save_dir)
        # save_images(
        #     out_fold=local_save_dir, fnames=[os.path.basename(p).format("normal") for p in paths],
        #     imgs=geo_normal.squeeze(0).detach().cpu().numpy()
        # )
        save_images(
            out_fold=local_save_dir, fnames=[os.path.basename(p).format(f"mask_{image_suffix}") for p in paths],
            imgs=mask_pred.squeeze(0).unsqueeze(1).detach().cpu().numpy()
        )
        save_images(
            out_fold=local_save_dir, fnames=[os.path.basename(p).format(f"mask_diff_{image_suffix}") for p in paths],
            imgs=mask_diff_img.squeeze(0).unsqueeze(1).detach().cpu().numpy()
        )
        save_images(
            out_fold=local_save_dir, fnames=[os.path.basename(p).format(f"shading_{image_suffix}") for p in paths],
            imgs=shading.squeeze(0).detach().cpu().numpy()
        )
        save_images(
            out_fold=local_save_dir, fnames=[os.path.basename(p).format(f"shading_bones_{image_suffix}") for p in paths],
            imgs=shading_with_bones.squeeze(0).detach().cpu().numpy()
        )
        save_images(
            out_fold=local_save_dir, fnames=[os.path.basename(p).format(f"rgb_overlayed_{image_suffix}") for p in paths],
            imgs=rgb_overlayed.squeeze(0).detach().cpu().numpy()
        )
        save_obj(
            out_fold=local_save_dir, fnames=[os.path.basename(p).format(f"mesh_{image_suffix}") for p in paths],
            meshes=shape.first_n(b0), feat=im_features[:b0], save_material=False
        )
        save_txt(
            out_fold=local_save_dir, fnames=[os.path.basename(p).format(f"keypoint_{image_suffix}") for p in paths],
            data=rearrange(keypoint_3d, "b f k d -> (b f) k d").cpu().numpy(), delim=' '
        )
        save_txt(
            out_fold=local_save_dir, fnames=[os.path.basename(p).format(f"eval_aux_{image_suffix}") for p in paths],
            data=eval_aux.cpu().numpy(), delim=' '
        )
        return



# Differentialble with respect to fovy
def get_camera_extrinsics_from_pose_differentiable(self, pose, znear=0.1, zfar=1000., offset_extra=None):
    def perspective(fovy, aspect=1.0, n=0.1, f=1000.0, device=None):
        y = torch.tan(fovy / 2)
        mat = torch.zeros((4, 4), dtype=torch.float32, device=device)
        mat[0, 0] = 1.0 / (y * aspect)
        mat[1, 1] = 1.0 / (-y)
        mat[2, 2] = -(f + n) / (f - n)
        mat[2, 3] = -(2 * f * n) / (f - n)
        mat[3, 2] = -1.0
        return mat
    N = len(pose)
    pose_R = pose[:, :9].view(N, 3, 3).transpose(2, 1)  # to be compatible with pytorch3d
    if offset_extra is not None:
        cam_pos_offset = torch.FloatTensor([0, 0, -self.cfg_pose.cam_pos_z_offset - offset_extra]).to(pose.device)
    else:
        cam_pos_offset = torch.FloatTensor([0, 0, -self.cfg_pose.cam_pos_z_offset]).to(pose.device)
    pose_T = pose[:, -3:] + cam_pos_offset[None, None, :]
    pose_T = pose_T.view(N, 3, 1)
    pose_RT = torch.cat([pose_R, pose_T], axis=2)  # Nx3x4
    w2c = torch.cat([pose_RT, torch.FloatTensor([0, 0, 0, 1]).repeat(N, 1, 1).to(pose.device)], axis=1)  # Nx4x4
    proj = perspective(self.fov / 180 * np.pi, 1, znear, zfar)[None].to(pose.device)  # assuming square images
    mvp = torch.matmul(proj, w2c)
    campos = -torch.matmul(pose_R.transpose(2, 1), pose_T).view(N, 3)
    return mvp, w2c, campos


def get_keypoint_coordinates(bones, mvp):
    """
    Input posed bones in world space and mvp matrix, output keypoint coordinates in clip space
    head -> tail -> LF -> RF -> LR -> RR
    Following Animal3D, normalize xy coordinate but keep z(depth) unormalized
    """
    keypoint_mapping = {  # keypoint index : (bone_index, bone_end_index(0 or 1))
        0: [(0,1)], 1: [(0,0),(1,1)], 2: [(1,0),(2,1)], 3: [(2,0),(3,1),(10,0),(19,0)],  # head -> mid
        4: [(3,0), (7,0)],  # mid
        5: [(6,0),(7,1)], 6: [(5,0),(6,1),(13,0),(16,0)], 7: [(4,0),(5,1)], 8: [(4,1)],  # mid -> tail
        9: [(9,0),(10,1)], 10: [(8,0),(9,1)], 11: [(8,1)],   # LF
        12: [(18,0),(19,1)], 13: [(17,0),(18,1)], 14: [(17,1)],  # RF
        15: [(12,0),(13,1)], 16: [(11,0),(12,1)], 17: [(11,1)],  # LR
        18: [(15,0),(16,1)], 19: [(14,0),(15,1)], 20: [(14,1)],  # RR
    }
    b, f, k, v = bones.shape[:4]
    bones = rearrange(bones, "b f k v d -> (b f) (k v) d")
    assert bones.shape[0] == mvp.shape[0] == b * f
    bones_clip4 = xfm_points(bones, mvp)
    bones_clip4 /= bones_clip4[..., 3:4]
    bones_clip4 = rearrange(bones_clip4, "(b f) (k v) d -> b f k v d", b=b, f=f, k=k, v=v)
    keypoint = torch.zeros((b, f, 21, 4), device=bones.device)
    for k, v in keypoint_mapping.items():
        for bone_idx in v:
            keypoint[:, :, k, :] += bones_clip4[:, :, bone_idx[0], bone_idx[1], :]
        keypoint[:, :, k, :] /= len(v)
    return keypoint


def get_eval_aux(shape, mvp, resolution=(256, 256)):
    """
    Save all vertices in clip space project to screen, with visibility mask (..., 4)
    KT evaluation will map source gt keypoint to the nearest visible vertex in 2D and transfer to same vertex in target 2D
    PCK evaluation will later learn mapping 3D shape -> 3D keypoint -> 2D keypoint
    """
    visibility = torch.zeros((shape.v_pos.shape[0], shape.v_pos.shape[1]), device=shape.v_pos.device, dtype=torch.bool)
    v_pos_clip4 = xfm_points(shape.v_pos, mvp)
    v_pos_clip4 = v_pos_clip4 / v_pos_clip4[..., 3:]
    rast, _ = rasterize(RasterizeGLContext(), v_pos_clip4, shape.t_pos_idx[0].int(), resolution)
    face_ids = rast[..., -1]
    for i, (face_id, face) in enumerate(zip(face_ids, shape.t_pos_idx)):
        visible_faces = torch.unique(face_id)
        visible_faces = visible_faces[(visible_faces >= 0) & (visible_faces < face.shape[0])].long()
        visible_vertices = torch.unique(face[visible_faces])
        visibility[i][visible_vertices] = True
    eval_aux = torch.cat([v_pos_clip4[..., :3], visibility.unsqueeze(-1)], dim=-1)
    return eval_aux
