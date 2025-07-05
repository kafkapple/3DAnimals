import torch
import numpy as np
from einops import rearrange

from model.utils import misc
from model.render import mesh
from model.geometry.skinning import skinning
from model.predictors.InstancePredictorFauna import InstancePredictorFauna, FaunaInstancePredictorConfig



class InstancePredictorFauna4D(InstancePredictorFauna):
    def __init__(self, cfg: FaunaInstancePredictorConfig):
        super().__init__(cfg)
        misc.load_cfg(self, cfg, FaunaInstancePredictorConfig)
        self.articulation_dict = TensorDict()
        self.pose_dict = TensorDict()

    def forward_deformation(self, shape, feat=None, batch_size=None, num_frames=None):
        original_verts = shape.v_pos
        num_verts = original_verts.shape[1]
        deform_feat = None
        if feat is not None:
            deform_feat = feat[:, None, :].repeat(1, num_verts, 1)  # Shape: (B, num_verts, latent_dim)
            original_verts = original_verts.repeat(len(feat), 1, 1)
        deformation = self.netDeform(original_verts, deform_feat) * 0.1  # Shape: (B, num_verts, 3), multiply by 0.1 to minimize disruption when initially enabled
        # if deformation.shape[0] > 1 and self.cfg_deform.force_avg_deform:
        #     assert batch_size is not None and num_frames is not None
        #     assert deformation.shape[0] == batch_size * num_frames
        #     deformation = deformation.view(batch_size, num_frames, *deformation.shape[1:])
        #     deformation = deformation.mean(dim=1, keepdim=True)
        #     deformation = deformation.repeat(1,num_frames,*[1]*(deformation.dim()-2))
        #     deformation = deformation.view(batch_size*num_frames, *deformation.shape[2:])
        shape = shape.deform(deformation)
        return shape, deformation

    def forward_pose(self, patch_out, patch_key, frame_ids=None, **kwargs):
        # Add netPose prediction to articulation_dict if not in dict
        with torch.no_grad():
            if self.cfg_pose.architecture == 'encoder_dino_patch_key':
                pose_gt = self.netPose(patch_key)  # Shape: (B, latent_dim)
            elif self.cfg_pose.architecture == 'encoder_dino_patch_out':
                pose_gt = self.netPose(patch_out)  # Shape: (B, latent_dim)
            else:
                raise NotImplementedError
        frame_ids = rearrange(frame_ids, "b f -> (b f)")
        self.pose_dict[frame_ids] = pose_gt  # no-op if frame_ids already in dict
        pose = self.pose_dict(frame_ids)

        ## xyz translation
        trans_pred = pose[..., -3:].tanh() * self.max_trans_xyz_range.to(pose.device)

        ## rotation
        if self.cfg_pose.rot_rep == 'euler_angle':
            rot_pred = pose[..., :3].tanh() * self.max_rot_xyz_range.to(pose.device)

        elif self.cfg_pose.rot_rep == 'quaternion':
            quat_init = torch.FloatTensor([0.01, 0, 0, 0]).to(pose.device)
            rot_pred = pose[..., :4] + quat_init
            rot_pred = torch.nn.functional.normalize(rot_pred, p=2, dim=-1)
            # rot_pred = torch.cat([rot_pred[...,:1].abs(), rot_pred[...,1:]], -1)  # make real part non-negative
            rot_pred = rot_pred * rot_pred[..., :1].sign()  # make real part non-negative

        elif self.cfg_pose.rot_rep == 'lookat':
            vec_forward = pose[..., :3]
            if self.cfg_pose.lookat_zeroy:
                vec_forward = vec_forward * torch.FloatTensor([1, 0, 1]).to(pose.device)
            vec_forward = torch.nn.functional.normalize(vec_forward, p=2, dim=-1)  # x right, y up, z forward
            rot_pred = vec_forward

        elif self.cfg_pose.rot_rep in ['quadlookat', 'octlookat']:
            rots_pred = pose[..., :self.num_pose_hypos * 4].view(-1, self.num_pose_hypos, 4)  # (B*F, K, 4)
            rots_logits = rots_pred[..., :1]
            vec_forward = rots_pred[..., 1:4]

            def softplus_with_init(x, init=0.5):
                assert np.abs(init) > 1e-8, "initial value should be non-zero"
                beta = np.log(2) / init
                return torch.nn.functional.softplus(x, beta=beta)

            xs, ys, zs = vec_forward.unbind(-1)
            xs = softplus_with_init(xs, init=0.5)  # initialize to 0.5
            if self.cfg_pose.rot_rep == 'octlookat':
                ys = softplus_with_init(ys, init=0.5)  # initialize to 0.5
            if self.cfg_pose.lookat_zeroy:
                ys = ys * 0
            zs = softplus_with_init(zs, init=0.5)  # initialize to 0.5
            vec_forward = torch.stack([xs, ys, zs], -1)
            vec_forward = vec_forward * self.orthant_signs.to(pose.device)
            vec_forward = torch.nn.functional.normalize(vec_forward, p=2, dim=-1)  # x right, y up, z forward
            rot_pred = torch.cat([rots_logits, vec_forward], -1).view(-1, self.num_pose_hypos * 4)  # (B*F, K*4)

        else:
            raise NotImplementedError

        pose = torch.cat([rot_pred, trans_pred], -1)
        return pose

    def forward_articulation(self, shape, feat, patch_feat, mvp, w2c, batch_size, num_frames, epoch, total_iter, frame_ids=None, **kwargs):
        """
        Inherited from InstancePredictorBase, additionally take frame_ids for arti params reconstruction
        """
        verts = shape.v_pos
        if len(verts) == batch_size * num_frames:
            verts = verts.view(batch_size, num_frames, *verts.shape[1:])  # BxFxNx3
        else:
            verts = verts[None]  # 1x1xNx3

        bones, bones_feat, bones_pos_in = self.get_bones(
            verts, feat, patch_feat, mvp, w2c, batch_size, num_frames, epoch, total_iter
        )

        # forward motion reconstruction using frame_ids get pred articulation angles
        b, f = frame_ids.shape
        frame_ids = rearrange(frame_ids, "b f -> (b f)")
        # Add netArticulation prediction to articulation_dict if not in dict
        with torch.no_grad():
            articulation_angles_out = self.netArticulation(bones_feat, bones_pos_in).view(batch_size, num_frames, bones.shape[2], 3)
            articulation_angles_gt = self.cfg_articulation.output_multiplier * articulation_angles_out.clone().detach()
            articulation_angles_gt = articulation_angles_gt.tanh()
            articulation_angles_gt = self.apply_articulation_constraints(articulation_angles_gt, total_iter)
            self.articulation_angles_gt = self.apply_fauna_articulation_regularizer(articulation_angles_gt, total_iter)
        articulation_angles_out = rearrange(articulation_angles_out, "b f n c -> (b f) n c")
        self.articulation_dict[frame_ids] = articulation_angles_out  # no-op if frame_ids already in dict
        articulation_angles_pred = self.articulation_dict(frame_ids)
        articulation_angles_pred = rearrange(articulation_angles_pred, "(b f) n c -> b f n c", b=b, f=f)
        articulation_angles_pred = self.cfg_articulation.output_multiplier * articulation_angles_pred
        articulation_angles_pred = articulation_angles_pred.tanh()
        articulation_angles_pred = self.apply_articulation_constraints(articulation_angles_pred, total_iter)
        articulation_angles_pred = self.apply_fauna_articulation_regularizer(articulation_angles_pred, total_iter)
        self.articulation_angles_pred = articulation_angles_pred

        # skinning and make pred shape
        verts_articulated_pred, aux = skinning(
            verts, bones, self.kinematic_tree, articulation_angles_pred, output_posed_bones=True,
            temperature=self.cfg_articulation.skinning_temperature
        )
        verts_articulated_pred = verts_articulated_pred.view(batch_size * num_frames, *verts_articulated_pred.shape[2:])
        v_tex = shape.v_tex
        if len(v_tex) != len(verts_articulated_pred):
            v_tex = v_tex.repeat(len(verts_articulated_pred), 1, 1)
        articulated_shape_pred = mesh.make_mesh(
            verts_articulated_pred, shape.t_pos_idx, v_tex, shape.t_tex_idx, shape.material
        )
        return articulated_shape_pred, articulation_angles_pred, aux


class TensorDict(torch.nn.Module):
    """
    Custom tensor dictionary to store index-tensor mapping as key-value pairs
    """
    def __init__(self):
        super().__init__()
        self.tensor_dict = torch.nn.ParameterDict()
        self.previous_tensor_mean_dict = {}
        self.leg_bone_idx = [8,9,10,11,12,13,14,15,16,17,18,19]

    def __setitem__(self, keys, values):
        assert keys.shape[0] == values.shape[0]
        for key, value in zip(keys, values):
            key = str(int(key.item()))
            if key in self.tensor_dict:
                continue
            else:
                arti_params = value.clone().detach()
                # arti_params[self.leg_bone_idx] = torch.randn_like(arti_params[self.leg_bone_idx])
                self.tensor_dict[key] = torch.nn.Parameter(arti_params, requires_grad=True)
                self.previous_tensor_mean_dict[key] = value.mean().item()

    def __getitem__(self, keys):
        for key in keys:
            key = str(int(key.item()))
            prev_mean = self.previous_tensor_mean_dict[key]
            curr_mean = self.tensor_dict[key].mean().item()
            if prev_mean != curr_mean:
                # print(f"TensorDict: {key} has changed, diff={curr_mean-prev_mean}", flush=True)
                self.previous_tensor_mean_dict[key] = self.tensor_dict[key].mean().item()
            else:
                # print(f"TensorDict: {key} has not changed", flush=True)
                pass
        return torch.stack([self.tensor_dict[str(int(key.item()))] for key in keys], dim=0)

    def items(self):
        return self.tensor_dict.items()

    def forward(self, indices):
        return self.__getitem__(indices)