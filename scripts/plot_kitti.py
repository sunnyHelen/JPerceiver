import copy
from matplotlib import pyplot as plt
import numpy as np
import os
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seq", type=str, default="10")
parser.add_argument("--outpath", type=str, default="VOpics")
parser.add_argument("--align", type=str, default="scale", choices=["scale", "scale_7dof", "7dof", "6dof"])
parser.add_argument("--type", type=str, default="learning", choices=["learning", "geometry", "infovo"])


def scale_lse_solver(X, Y):
    """Least-sqaure-error solver
    Compute optimal scaling factor so that s(X)-Y is minimum
    Args:
        X (KxN array): current data
        Y (KxN array): reference data
    Returns:
        scale (float): scaling factor
    """
    scale = np.sum(X * Y)/np.sum(X ** 2)
    return scale

def scale_optimization(gt, pred):
    """ Optimize scaling factor
    Args:
        gt (4x4 array dict): ground-truth poses
        pred (4x4 array dict): predicted poses
    Returns:
        new_pred (4x4 array dict): predicted poses after optimization
    """
    pred_updated = copy.deepcopy(pred)
    xyz_pred = []
    xyz_ref = []
    for i in pred:
        pose_pred = pred[i]
        pose_ref = gt[i]
        xyz_pred.append(pose_pred[:3, 3])
        xyz_ref.append(pose_ref[:3, 3])
    xyz_pred = np.asarray(xyz_pred)
    xyz_ref = np.asarray(xyz_ref)
    scale = scale_lse_solver(xyz_pred, xyz_ref)
    for i in pred_updated:
        pred_updated[i][:3, 3] *= scale
    return pred_updated

def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        assert False, "x.shape not equal to y.shape"

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


def load_poses_from_txt(file_name):
    """Load poses from txt (KITTI format)
    Each line in the file should follow one of the following structures
        (1) idx pose(3x4 matrix in terms of 12 numbers)
        (2) pose(3x4 matrix in terms of 12 numbers)

    Args:
        file_name (str): txt file path
    Returns:
        poses (dict): {idx: 4x4 array}
    """
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    poses = {}
    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ") if i!=""]
        withIdx = len(line_split) == 13
        # assert not withIdx
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row*4 + col + withIdx]
        if withIdx:
            frame_idx = line_split[0]
        else:
            frame_idx = cnt
        poses[frame_idx] = P
    return poses


def plot_trajectory(all_poses, seq, colors, type, align, outpath="."):
    """Plot trajectory for both GT and prediction
    Args:
        poses_gt (dict): {idx: 4x4 array}; ground truth poses
        poses_result (dict): {idx: 4x4 array}; predicted poses
        seq (int): sequence index.
    """
    plot_keys = all_poses.keys()
    fontsize_ = 60
    if seq == "09":
        fig_size = 15
    if seq in ["05", "07", "10"]:
        fig_size = 50

    fig = plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    # plt.draw()
    # plt.axis('scaled')

    for key in plot_keys:
        pos_xz = []
        frame_idx_list = sorted(all_poses[key].keys())
        for frame_idx in frame_idx_list:
            # pose = np.linalg.inv(poses_dict[key][frame_idx_list[0]]) @ poses_dict[key][frame_idx]
            pose = all_poses[key][frame_idx]
            pos_xz.append([pose[0, 3],  pose[2, 3]])
        pos_xz = np.asarray(pos_xz)
        plt.plot(pos_xz[:, 0],  pos_xz[:, 1], colors[key], label=key, linewidth=15.0)

    plt.legend(loc="upper left", prop={'size': fontsize_})
    plt.xticks(fontsize=fontsize_)
    plt.yticks(fontsize=fontsize_)
    ax = plt.gca()
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    plt.xlabel('x(m)', fontsize=fontsize_)
    plt.ylabel('z(m)', fontsize=fontsize_)
    fig.set_size_inches(fig_size, fig_size)
    # fig.set_figheight(20)
    # fig.set_figwidth(20)
    png_title = "seq_{}_{}_{}".format(seq, type, align)

    fig_pdf = outpath + "/" + png_title + ".png"
    # plt.show()
    plt.savefig(fig_pdf, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close(fig)


def plot_kitti(opt, results):
    """
    results -> dict, {"GT": "../gt_pose.txt"}
    """
    seq = opt.seq
    alignment = opt.align

    assert "GT" in results.keys()
    poses_gt = load_poses_from_txt(results["GT"][0])

    # Pose alignment to first frame
    idx_0 = sorted(list(poses_gt.keys()))[0]
    gt_0 = poses_gt[idx_0]
    for cnt in poses_gt:
        poses_gt[cnt] = np.linalg.inv(gt_0) @ poses_gt[cnt]
    
    # Create the final trajectory dict
    all_poses = {
        "GT": poses_gt
    }

    colors = {}
    
    # Save xyz for GT for later alignment usage
    # Error: Since pose_results's indices might be different than ground truth
    if alignment == "scale_7dof" or alignment == "7dof" or alignment == "6dof":
        xyz_gt = []
        for cnt in poses_gt:
            xyz_gt.append([poses_gt[cnt][0, 3], poses_gt[cnt][1, 3], poses_gt[cnt][2, 3]])
        xyz_gt = np.asarray(xyz_gt).transpose(1, 0)

    # Read pose files other than GT
    for exp, pose_file in results.items():
        colors[exp] = pose_file[1]
        if exp == "GT": 
            continue 
        poses_result = load_poses_from_txt(pose_file[0])
        idx_0 = sorted(list(poses_result.keys()))[0]
        pred_0 = poses_result[idx_0]
        for cnt in poses_result:
            poses_result[cnt] = np.linalg.inv(pred_0) @ poses_result[cnt]
        
        if alignment == "scale":
            poses_result = scale_optimization(poses_gt, poses_result)
        elif alignment == "scale_7dof" or alignment == "7dof" or alignment == "6dof":
            xyz_gt = []
            xyz_result = []
            for cnt in poses_result:
                xyz_gt.append([poses_gt[cnt][0, 3], poses_gt[cnt][1, 3], poses_gt[cnt][2, 3]])
                xyz_result.append([poses_result[cnt][0, 3], poses_result[cnt][1, 3], poses_result[cnt][2, 3]])
            xyz_gt = np.asarray(xyz_gt).transpose(1, 0)
            xyz_result = np.asarray(xyz_result).transpose(1, 0)

            r, t, scale = umeyama_alignment(xyz_result, xyz_gt, alignment!="6dof")

            align_transformation = np.eye(4)
            align_transformation[:3:, :3] = r
            align_transformation[:3, 3] = t

            for cnt in poses_result:
                poses_result[cnt][:3, 3] *= scale
                if alignment=="7dof" or alignment=="6dof":
                    poses_result[cnt] = align_transformation @ poses_result[cnt]
        all_poses[exp] = poses_result
    
    # Now plot the trajectories in all_poses
    plot_trajectory(all_poses, seq, colors, opt.type,opt.align, opt.outpath)
            


if __name__=="__main__":
    # NOTE: Define the results to display here
    # NOTE: opt: --seq --align --outpath
    opt = parser.parse_args()
    basedir = "/CV/users/zhaohaimei3/DepthLayout_kitti_odometry_loss_transformerright_object_argoverse_A100-1024/scripts/"
    if opt.type == "learning":
        results = {
            "GT": [basedir + "poses/{}.txt".format(opt.seq), "black"],
            # "GT": ["ground-truth/{}.txt".format(opt.seq), "black"],
            "Ours": [basedir + "ours/{}.txt".format(opt.seq), "red"],
            # "DF-VO (M-Train)": ["df-vo/mono_sc/{}.txt".format(opt.seq), "limegreen"],
             "Monodepth2": [basedir + "other/baseline/{}.txt".format(opt.seq), "darkgreen"],
            "SC-SfMLearner": [basedir + "other/sc-sfmlearner/{}.txt".format(opt.seq), "blue"],
            "SfMLearner": [basedir + "other/sfmlearner/{}.txt".format(opt.seq), "purple"],
            #"DNet": [basedir + "other/DNet/{}.txt".format(opt.seq), "darkgoldenrod"]
        }
    
    elif opt.type == "geometry":
        results = {
            "GT": ["ground-truth/{}.txt".format(opt.seq),  "black"],
            "Ours": ["ours/hpc-0710d1/{}.txt".format(opt.seq), "crimson"],
            "DSO": ["other/dso/{}.txt".format(opt.seq), "saddlebrown"],
            "VISO2": ["other/viso2/{}.txt".format(opt.seq), "limegreen"],
            "ORB-SLAM2 (w/o LC)": ["other/orbslam2/KITTI_without_lc/best_ate/{}.txt".format(opt.seq), "darkorange"],
            "ORB-SLAM2 (w/ LC)": ["other/orbslam2/KITTI_with_lc/best_ate/{}.txt".format(opt.seq), "cornflowerblue"]
        }
    
    elif opt.type == "infovo":
        results = {
            "GT": ["ground-truth/{}.txt".format(opt.seq),  "black"],
            "DeepVO": ["info-odo/deepvo/abs/{}.txt".format(opt.seq), "crimson"],
            "InfoVO": ["info-odo/infovo/abs/{}.txt".format(opt.seq), "cornflowerblue"],
            "VINet": ["info-odo/vinet/abs/{}.txt".format(opt.seq), "limegreen"],
            "InfoVIO": ["info-odo/infovio/abs/{}.txt".format(opt.seq), "darkorange"],
            # "VISO2": ["other/viso2/{}.txt".format(opt.seq), "purple"],
            # "ORB-SLAM2 (w/o LC)": ["other/orbslam2/KITTI_without_lc/best_ate/{}.txt".format(opt.seq), "cornflowerblue"],
            # "ORB-SLAM2 (w/ LC)": ["other/orbslam2/KITTI_with_lc/best_ate/{}.txt".format(opt.seq), "purple"]
            # "Soft-Fusion": ["info-odo/soft/abs/{}.txt".format(opt.seq), "cornflowerblue"],
            # "Hard-Fusion": ["info-odo/hard/abs/{}.txt".format(opt.seq), "purple"]
        }

    plot_kitti(opt, results)









