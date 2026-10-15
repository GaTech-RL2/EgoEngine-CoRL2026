import os
from os.path import join, abspath, dirname, isfile
import numpy as np
import pickle
import torch
from manopth.manolayer import ManoLayer
# import open3d as o3d
import trimesh 

def read_hand_shape(hand_shape_path):
    assert hand_shape_path.endswith(".pkl")
    if not isfile(hand_shape_path):
        return np.zeros(10)
    beta = pickle.load(open(hand_shape_path, "rb"))["hand_shape"]
    beta = np.float32(beta)
    return beta

def estimate_local_frame(tip_name, path_names, name_ord, prev_poses, is_rhand):
    """
    Re-computes a single fingertip’s 3×3 rotation from its wrist+3 mid-joints.
    prev_poses: (21×4×4) numpy array of current joint SE(3)s 
    returns: new 21×4×4 numpy array with only that tip’s R replaced.
    """
    new_poses = prev_poses.copy()
    tip_id    = name_ord.index(tip_name)
    wst_id    = name_ord.index("wrist")
    tip_pos   = prev_poses[tip_id, :3, 3]
    wst_pos   = prev_poses[wst_id, :3, 3]

    # 1) compute averaged normal n
    normals = []
    for mid_name in path_names:
        mid_id  = name_ord.index(mid_name)
        mid_pos = prev_poses[mid_id, :3, 3]
        v1      = wst_pos - mid_pos
        v2      = mid_pos - tip_pos
        n_vec   = np.cross(v2, v1)
        normals.append(n_vec / (np.linalg.norm(n_vec) + 1e-9))
    n = np.mean(normals, axis=0)
    n /= (np.linalg.norm(n) + 1e-9)

    # 2) compute approach vector v
    first_mid_id = name_ord.index(path_names[0])
    v = tip_pos - prev_poses[first_mid_id, :3, 3]
    v /= (np.linalg.norm(v) + 1e-9)

    # 3) pick axes
    if "thumb" in tip_name:
        z_vec = v
        y_vec = -n
        x_vec = np.cross(y_vec, z_vec)
    else:
        z_vec = n
        x_vec = v
        y_vec = np.cross(z_vec, x_vec)

    # 4) right-hand sign flips
    if is_rhand:
        if "thumb" in tip_name:
            x_vec, y_vec = -x_vec, -y_vec
        else:
            x_vec, z_vec = -x_vec, -z_vec

    # 5) normalize
    x_vec /= (np.linalg.norm(x_vec) + 1e-9)
    y_vec /= (np.linalg.norm(y_vec) + 1e-9)
    z_vec /= (np.linalg.norm(z_vec) + 1e-9)

    new_rot = np.stack([x_vec, y_vec, z_vec], axis=1)  # 3×3
    new_poses[tip_id, :3, :3] = new_rot
    return new_poses


def mano_params_to_hand_info(
        hand_pose_path,
        mano_beta=None,
        side="right",
        max_cnt=None,
        return_pose=False,
        return_faces=False,
        return_frames=False,
        device="cuda:0"):
    """
    If `return_frames` is True an extra tensor is returned:
        hand_se3  –  shape (N, 21, 4, 4) – full 6‑DoF pose for
                     wrist + 20 joints, in metre units.
    """

    # ** original MANO joint order (right hand)
    #                16-15-14-13-\
    #                             \
    #          17 --3 --2 --1------0
    #        18 --6 --5 --4-------/
    #        19 -12 -11 --10-----/
    #          20 --9 --8 --7---/

    #  transform order of right hand
    #         15-14-13-\
    #                   \
    #    3-- 2 -- 1 -----0   < NOTE: demo on this finger
    #   6 -- 5 -- 4 ----/
    #   12 - 11 - 10 --/
    #    9-- 8 -- 7 --/

    # ---------- 1. Load MANO parameters and get basic geometry using manopth ----------
    betas = torch.from_numpy(mano_beta).unsqueeze(0).float().to(device)
    mano_layer = ManoLayer(
        mano_root=join(dirname(dirname(abspath(__file__))), 'manopth/mano/models'),
        use_pca=False, ncomps=45, side=side, center_idx=0).to(device)

    theta_list, trans_list = [], []
    if hand_pose_path.endswith(".pkl"):
        data = pickle.load(open(hand_pose_path, "rb"))
        for k in sorted(data.keys()):
            theta_list.append(data[k]["hand_pose"].cpu().numpy())
            trans_list.append(data[k]["hand_trans"].cpu().numpy())
    else:
        fns = sorted(os.listdir(hand_pose_path))[:max_cnt] if max_cnt else sorted(os.listdir(hand_pose_path))
        for fn in fns:
            d = pickle.load(open(join(hand_pose_path, fn), "rb"))
            theta_list.append(d["hand_pose"].cpu().numpy())
            trans_list.append(d["hand_trans"].cpu().numpy())

    batch_theta = torch.from_numpy(np.float32(theta_list)).to(device)
    batch_trans = torch.from_numpy(np.float32(trans_list)).to(device)

    betas = betas.repeat(batch_theta.shape[0], 1)
    verts_pred, joints_pred = mano_layer(batch_theta, betas)
    verts_pred = verts_pred / 1000.0 + batch_trans.unsqueeze(1)
    joints_pred = joints_pred / 1000.0 + batch_trans.unsqueeze(1)

    hand_vertices = verts_pred.cpu().numpy()
    hand_joints = joints_pred.cpu().numpy()

    # ---------- 2. Build 6-DoF joint frames if requested ----------
    hand_se3 = None
    if return_frames:
        # Use manotorch, which easily exposes absolute transforms
        from manotorch.manolayer import ManoLayer as ManoLayerMT
        
        # NOTE: Make sure this path is correct for your system
        mano_assets_path = join(dirname(abspath(__file__)), 'manotorch/assets/mano')
        if not os.path.exists(mano_assets_path):
             # Fallback for a different project structure
            mano_assets_path = join(dirname(dirname(abspath(__file__))), "manotorch/mano")
        
        mt_layer = ManoLayerMT(
            use_pca=False, side=side, ncomps=45, flat_hand_mean=False,
            mano_assets_root=mano_assets_path
        ).to(device)

        mt_out = mt_layer(batch_theta, betas)
        
        # Get absolute rotations for the 16 articulated joints
        rotations_abs = mt_out.transforms_abs[:, :, :3, :3] # Shape: (N, 16, 3, 3)
        
        # We already have the final world positions for all 21 joints
        final_joint_positions = torch.from_numpy(hand_joints).to(device) # Shape: (N, 21, 3)

        # Reordering indices to match the original script's joint structure
        idx0 = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        idx1 = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
        def _get_mano_idx(_idx):
            snap2mano = dict(zip(range(len(_idx)), _idx))
            mano2snap = {v: k for k, v in snap2mano.items()}
            return [mano2snap[k] for k in range(len(_idx))]
        r_idx0 = _get_mano_idx(idx0)
        r_idx1 = _get_mano_idx(idx1)

        # --- FIX: Correctly assemble the final SE(3) pose matrices ---
        se3 = torch.eye(4, device=device).repeat(final_joint_positions.shape[0], 21, 1, 1)
        
        # 1. Place the 16 absolute rotation matrices into the correct slots of the 21-joint structure
        se3[:, r_idx0, :3, :3] = rotations_abs
        
        # 2. Set the translation for all 21 joints using the final world positions
        se3[:, :, :3, 3] = final_joint_positions[:, r_idx1]
        
        # --- 3. Fingertip orientation calculation (same logic as before) ---
        links = {16:[15,14,13], 17:[3,2,1], 18:[6,5,4], 19:[12,11,10], 20:[9,8,7]}
        for tip_idx_orig, chain_indices_orig in links.items():
            tip_idx_new   = r_idx1[tip_idx_orig]
            chain_new     = [r_idx1[i] for i in chain_indices_orig]
            wrist_idx_new = r_idx1[0]

            tip   = se3[:, tip_idx_new,   :3, 3]          # (N,3)
            wrist = se3[:, wrist_idx_new, :3, 3]          # (N,3)
            mids  = se3[:, chain_new,     :3, 3]          # (N,3,3)

            v2 = mids - tip.unsqueeze(1)                  # mid − tip
            v1 = wrist.unsqueeze(1) - mids                # wrist − mid
            n  = torch.cross(v2, v1).mean(1)              # normal
            n  = n / (n.norm(dim=1, keepdim=True) + 1e-9)

            v  = tip - mids[:, 0]                         # tip − first‑mid
            v  = v / (v.norm(dim=1, keepdim=True) + 1e-9)

            if tip_idx_orig == 16:                        # thumb tip
                z = v
                y = -n
                x = torch.cross(y, z)
            else:                                         # other fingertips
                z = n
                x = v
                y = torch.cross(z, x)

            if side == "right":
                if tip_idx_orig == 16:                    # right thumb
                    x, y = -x, -y
                else:                                     # other right fingers
                    x, z = -x, -z

            # normalise axes
            x = x / (x.norm(dim=1, keepdim=True) + 1e-9)
            y = y / (y.norm(dim=1, keepdim=True) + 1e-9)
            z = z / (z.norm(dim=1, keepdim=True) + 1e-9)

            se3[:, tip_idx_new, :3, :3] = torch.stack([x, y, z], -1)

        hand_se3 = se3.cpu().numpy()

    se3_np = se3.cpu().numpy()
    mano_joint_names = [ 
        "wrist","index1","index2","index3",
        "middle1","middle2","middle3",
        "pinky1","pinky2","pinky3",
        "ring1","ring2","ring3",
        "thumb1","thumb2","thumb3",
        "thumb_tip","index_tip","middle_tip","ring_tip","pinky_tip"
    ]
    links = {
    "thumb_tip":  ["thumb3","thumb2","thumb1"],
    "index_tip":  ["index3","index2","index1"],
    "middle_tip": ["middle3","middle2","middle1"],
    "ring_tip":   ["ring3","ring2","ring1"],
    "pinky_tip":  ["pinky3","pinky2","pinky1"],
    }

    B = se3_np.shape[0]
    for b in range(B):
        for tip_name, path_names in links.items():
            se3_np[b] = estimate_local_frame(
                tip_name, path_names, mano_joint_names,
                se3_np[b], is_rhand=(side=="right")
            )

    # rotate pose for robot wrist
    if side == "right":
        h2r_base = np.array([[0,0,1],
                        [0,-1,0],
                        [1,0,0]], dtype=np.float32)
    else:
        h2r_base = np.array([[0,0,1],
                        [0, 1,0],
                        [-1,0,0]], dtype=np.float32)
    inv_h2r = np.linalg.inv(h2r_base)

    # now do the final hand→robot base‐frame rotation
    se3_np[:, :, :3, :3] = se3_np[:, :, :3, :3] @ inv_h2r[None,None,:,:]
    hand_se3 = se3_np




    # ---------- Assemble return tuple ----------
    results = [hand_vertices, hand_joints]
    if return_frames:
        results.append(hand_se3)
    if return_faces:
        results.append(mano_layer.th_faces.cpu().numpy())
    
    if len(results) == 4:
        # Reorder to match the expected output of (verts, joints, frames, faces)
        return results[0], results[1], results[2], results[3]
    return tuple(results)

if __name__ == "__main__":
    ml = ManoLayer(mano_root=join(dirname(abspath(__file__)), 'manopth/mano/models'), use_pca=False, ncomps=45, side="right", center_idx=0)
    verts, _, _ = ml(torch.zeros(1, 48), torch.zeros(1, 10))
    verts = verts[0] / 1000.0
    print(verts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts.detach().cpu().numpy())
    o3d.io.write_point_cloud("./TPOSE.ply", pcd)