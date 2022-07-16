from __future__ import absolute_import, division, print_function
import os
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
from mmcv import Config
import glob
import torch
import io
from torchvision import transforms
from torch.utils.data import DataLoader
import PIL.Image as pil
sys.path.append('.')
sys.path.append('..')
from mono.model.registry import MONO
from mono.model.mono_baseline.layers import disp_to_depth
from mono.datasets.utils import readlines
from mono.datasets.kitti_dataset import KITTIRAWDataset
from collections import namedtuple
from argoverse.data_loading.simple_track_dataloader import SimpleArgoverseTrackingDataLoader
from argoverse.utils.calibration import get_calibration_config
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

MIN_DEPTH=1e-3
MAX_DEPTH=80
SCALE = 36#we set baseline=0.0015m which is 36 times smaller than the actual value (0.54m)
def get_static(path):
    tv = pil_loader(path)
    return tv.convert('L')
def pil_loader(path):
    with open(path, 'rb') as f:
        with pil.open(f) as img:
            return img.convert('RGB')
# def transform(cv2_img, height=1024, width=1024):
#     im_tensor = torch.from_numpy(cv2_img.astype(np.float32)).cuda().unsqueeze(0)
#     im_tensor = im_tensor.permute(0, 3, 1, 2).contiguous()
#     im_tensor = torch.nn.functional.interpolate(im_tensor, [height, width],mode='bilinear', align_corners=False)
#     im_tensor /= 255
#     return im_tensor
def transform(cv2_img, height=1024, width=1024):
    im_tensor = cv2_img.resize((width, height), pil.LANCZOS)
    im_tensor = transforms.ToTensor()(im_tensor).unsqueeze(0).cuda()

    return im_tensor
def get_intrinsic(img_path):

    # split_data_dir = frame_index.rsplit("/")[1]
    split_data_dir = img_path.split('/stereo_front_left/')[0]
    log_id = split_data_dir.rsplit("/")[1]
    dl = SimpleArgoverseTrackingDataLoader(data_dir=split_data_dir, labels_dir=split_data_dir)
    calib_data = dl.get_log_calibration_data(log_id)
    camera_name = "stereo_front_left"
    camera_config = get_calibration_config(calib_data, camera_name)
    camera_SE3_egovehicle = camera_config.extrinsic
    # print("------------",K)
    return camera_SE3_egovehicle
def predict(prev_img, cv2_img, model):
    original_height, original_width = cv2_img.size
    im_tensor = transform(cv2_img)
    if prev_img is not None:
        im_tensor_prev = transform(prev_img)
    else:
        im_tensor_prev = im_tensor

    with torch.no_grad():
        input = {}
        input['color_aug', 0, 0] = im_tensor

        input['color_aug', -1, 0] = im_tensor_prev
        outputs = model(input)
    # print(outputs.keys())
    disp = outputs[("disp", 0, 0)]
    disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)
    min_disp = 1/MAX_DEPTH
    max_disp = 1/MIN_DEPTH
    depth = 1/(disp_resized.squeeze().cpu().numpy()*max_disp + min_disp) #* SCALE
    layout = outputs["topview"].squeeze().cpu().numpy()
        # if prev_img is not None:
    T_ = outputs[("cam_T_cam", 0, -1)].cpu().numpy()[0]


    # true_top_view = np.zeros((layout.shape[1], layout.shape[2]))
    # true_top_view[layout[1] > layout[0]] = 255
    return depth, disp_resized.squeeze().cpu().numpy(), layout, T_
def predict_object(prev_img, cv2_img, model):
    original_height, original_width = cv2_img.size
    im_tensor = transform(cv2_img)
    if prev_img is not None:
        im_tensor_prev = transform(prev_img)
    else:
        im_tensor_prev = im_tensor

    with torch.no_grad():
        input = {}
        input['color_aug', 0, 0] = im_tensor

        input['color_aug', -1, 0] = im_tensor_prev
        outputs = model(input)
    # print(outputs.keys())
        layout = outputs["topview"].squeeze().cpu().numpy()
        # if prev_img is not None:

    # true_top_view = np.zeros((layout.shape[1], layout.shape[2]))
    # true_top_view[layout[1] > layout[0]] = 255
    return layout
def save_topview( tv, name_dest_im):
    SegClass = namedtuple('SegClass', ['name', 'id', 'train_id', 'color'])
    classes = [
        SegClass('background', 0, 0, (0, 0, 0)),
        SegClass('vehicle', 1, 1, (0, 0, 255)),
        SegClass('road', 2, 2, (255, 255, 255)),
        SegClass('pedestrain', 3, 3, (255, 0, 0)), ]
    color2index = {}
    index2color = {}
    for obj in classes:
        idx = obj.train_id
        label = obj.name
        color = obj.color
        color2index[color] = idx
        index2color[idx] = color
    tv_np = tv#.squeeze().cpu().numpy()
    # true_top_view = np.zeros((tv_np.shape[0], tv_np.shape[1]))
    tv_np = np.argmax(tv_np, axis=0)
    # print("tv_np",tv_np.shape)
    a = tv_np[0,0]
    height, width = tv_np.shape
    true_top_view = np.zeros((tv_np.shape[0], tv_np.shape[1],3))
    l = []
    # idx_mat = np.zeros((height, width))
    for h in range(height):
        for w in range(width):
            idx = tv_np[h, w]
            # print(color)
            # print(color)
            try:
                color = index2color[idx]
                # if index == 1:
                #     print(index)
                true_top_view[h, w, :] = color
                if color not in l:
                    l.append(color)
            except:
                # no index, assign to void
                true_top_view[h, w,:] = (0, 0, 0)
    true_top_view = true_top_view.astype(np.uint8)
    im = pil.fromarray(true_top_view)
    dir_name = os.path.dirname(name_dest_im)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    im.save(name_dest_im)
    # cv2.imwrite(name_dest_im, true_top_view)

    print("Saved prediction to {}".format(name_dest_im))
def save_topview_object(tv,tv_object, name_dest_im):
    SegClass = namedtuple('SegClass', ['name', 'id', 'train_id', 'color'])
    classes = [
        SegClass('background', 0, 0, (0, 0, 0)),
        SegClass('road', 1, 1, (255, 255, 255)),
        SegClass('vehicle', 2, 2, (0, 0, 255)),
        SegClass('pedestrain', 3, 3, (255, 0, 0)), ]
    # classes_object = [
    #     SegClass('background', 0, 0, (0, 0, 0)),
    #     SegClass('vehicle', 1, 1, (0, 0, 255)),
    #     SegClass('road', 2, 2, (255, 255, 255)),
    #     SegClass('pedestrain', 3, 3, (255, 0, 0)), ]
    color2index = {}
    index2color = {}
    index2color_object = {}
    for obj in classes:
        idx = obj.train_id
        label = obj.name
        color = obj.color
        color2index[color] = idx
        index2color[idx] = color
    # for obj in classes_object:
    #     idx = obj.train_id
    #     label = obj.name
    #     color = obj.color
    #     color2index[color] = idx
    #     index2color_object[idx] = color
    tv_np = tv#.squeeze().cpu().numpy()
    # true_top_view = np.zeros((tv_np.shape[0], tv_np.shape[1]))
    tv_np = np.argmax(tv_np, axis=0)
    # print("tv_np",tv_np.shape)
    a = tv_np[0,0]
    height, width = tv_np.shape
    tv_np_object = tv_object#.squeeze().cpu().numpy()
    # true_top_view = np.zeros((tv_np.shape[0], tv_np.shape[1]))
    tv_np_object = np.argmax(tv_np_object, axis=0)

    true_top_view = np.zeros((tv_np.shape[0], tv_np.shape[1],3))
    l = []
    # idx_mat = np.zeros((height, width))
    for h in range(height):
        for w in range(width):

            idx = tv_np[h, w]
            idx_object = tv_np_object[h, w]
            if idx_object == 1:
                idx = 2

            # print(color)
            # print(color)
            try:
                color = index2color[idx]
                # color_object = index2color_object[idx_object]
                # if index == 1:
                #     print(index)
                true_top_view[h, w, :] = color
                if color not in l:
                    l.append(color)
                # true_top_view[h, w, :] = color_object
                # if color_object not in l:
                #     l.append(color_object)
            except:
                # no index, assign to void
                true_top_view[h, w,:] = (0, 0, 0)
    true_top_view = true_top_view.astype(np.uint8)
    im = pil.fromarray(true_top_view)
    dir_name = os.path.dirname(name_dest_im)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    im.save(name_dest_im)
    # # cv2.imwrite(name_dest_im, true_top_view)
    #
    # print("Saved prediction to {}".format(name_dest_im))
    return true_top_view
def evaluate(cfg_path, model_path, img_path, output_path, output_path2):
    cfg = Config.fromfile(cfg_path)
    cfg['model']['depth_pretrained_path'] = None
    cfg['model']['pose_pretrained_path'] = None
    cfg['model']['extractor_pretrained_path'] = None
    model = MONO.module_dict[cfg.model['name']](cfg.model)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.cuda()
    model.eval()
    prev_img = None
    model_object = MONO.module_dict[cfg.model['name']](cfg.model)
    checkpoint = torch.load(model_path_object)
    model_object.load_state_dict(checkpoint['state_dict'], strict=True)
    model_object.cuda()
    model_object.eval()
    orig = np.array([0, 0, 0, 1])
    pos_gt = [orig.copy()]
    pos_pred = [orig.copy()]
    record_dir = img_path
    sequence = img_path.split("sequences/")[1].split("/")[0]
    vid_name = 'sequence_{}_47.avi'.format(sequence)
    out = cv2.VideoWriter(vid_name, cv2.VideoWriter_fourcc(*'XVID'), 10.0, (608 + 224 + 224, 224 * 2))
    T_pred = []
    T_gt = np.eye(4)
    T_pr = np.eye(4)
    if os.path.isfile(img_path):
        # Only testing on a single image
        paths = [img_path]
        output_directory = os.path.dirname(img_path)
    elif os.path.isdir(img_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(img_path, '*.{}'.format("png")))
        output_directory = output_path
    else:
        raise Exception("Can not find args.image_path: {}".format(img_path))
    print("-> Predicting on {:d} test images".format(len(paths)))
    with torch.no_grad():


        for idx, image_path in enumerate(sorted(os.listdir(img_path))):
            image_path = os.path.join(img_path, image_path)
            cv2_img = pil.open(image_path).convert('RGB')
            # cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            input_image_resized_write = cv2_img.resize((608, 224), pil.LANCZOS)
            # if prev_img is None:
            depth, disp_resized, true_top_view, T_ = predict(prev_img, cv2_img, model)
            true_top_view_object = predict_object(prev_img, cv2_img, model_object)
            vmax = np.percentile(disp_resized, 95)
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_path2, "{}_disp.jpeg".format(output_name))
            # print(output_path)
            plt.imsave(output_path, disp_resized, cmap='magma', vmax=vmax)
            output_path_layout = os.path.join(output_directory, "{}_layout.jpeg".format(output_name))
            layout_object = save_topview_object(true_top_view, true_top_view_object, output_path_layout)
            layout_gt_name = image_path.replace("image_2/", "road_dense128/")
            try:
                layout_gt = get_static(layout_gt_name).resize((224, 224), pil.LANCZOS)
            except:
                layout_gt = np.zeros((224, 224))
            videolayoutpath = "./videolayout.png"
            videolayout = pil_loader(videolayoutpath).resize((224, 224), pil.LANCZOS)
            if prev_img is not None:
                T_pr = T_pr.dot(T_)
                T_pred += [T_]
                #             T_gt = T_gt.dot(np.linalg.inv(pose_gt[i-1]).dot(pose_gt[i]))
                #             pos_gt += [T_gt[:,-1]]
                pos_pred += [T_pr[:, -1]]
            # disp_np = disp.squeeze().cpu().numpy()
            disp_np = cv2.resize(disp_resized, (608, 224))
            vmax = np.percentile(disp_np, 95)
            output = np.zeros((224 * 2, 608 + 224 + 224, 3), dtype=np.uint8)
            buf = io.BytesIO()
            plt.imsave(buf, disp_np, cmap='magma', vmax=vmax)
            buf.seek(0)
            disp_ = np.array(pil.open(buf))[:, :, :3]
            buf.close()

            # layout = cv2.resize(layout, (512, 512))
            # buf = io.BytesIO()
            # cv2.rectangle(layout, (91, 172), (101, 512), (0, 255, 255), thickness=-1)
            # plt.imsave(buf, layout)
            # buf.seek(0)
            # layout_ = np.array(pil.open(buf))[:, :, :3]
            # buf.close()

            layout_object = cv2.resize(layout_object, (224, 224))
            buf = io.BytesIO()
            cv2.rectangle(layout_object, (109, 210), (119, 224), (0, 255, 255), thickness=-1)
            plt.imsave(buf, layout_object)
            buf.seek(0)
            layout_ = np.array(pil.open(buf))[:, :, :3]
            buf.close()

            # layout_gt = cv2.resize(layout_gt, (512, 512))
            buf = io.BytesIO()
            # cv2.rectangle(layout_gt, (91, 172), (101, 512), (0, 255, 255), thickness=-1)
            plt.imsave(buf, layout_gt)
            buf.seek(0)
            layout_gt_ = np.array(pil.open(buf))[:, :, :3]
            buf.close()

            pos_gt_ = np.array(pos_gt)
            plt.figure(figsize=(3.5, 3.5))
            plt.title("Visual odometry", fontsize=15)
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='datalim')
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            #     plt.plot(pos_gt_[:, 0], pos_gt_[:, 2], 'o-', label='gt')
            scale = 29.5
            pos_pred_ = np.array(pos_pred) * scale
            plt.plot(pos_pred_[:, 0], pos_pred_[:, 2], 'o-', label='pred')
            plt.scatter(pos_pred_[-1, 0], pos_pred_[-1, 2], color='r', s=100, zorder=10)
            plt.text(pos_pred_[-1, 0], pos_pred_[-1, 2], 'NOW', fontsize=15, zorder=20)
            plt.legend(loc=4, fontsize=15)
            buf = io.BytesIO()
            plt.savefig(buf, dpi=64)
            plt.close()
            buf.seek(0)
            vo = np.array(pil.open(buf))[:, :, :3]
            buf.close()
            # print(output.shape)
            output[:224, :608] = np.array(input_image_resized_write)
            output[224:, :608] = disp_
            output[:224, 608:608 + 224] = layout_
            output[224:, 608:608 + 224] = vo
            output[:224, -224:] = layout_gt_
            output[-224:, -224:] = videolayout
            out.write(output[:, :, ::-1])

            prev_img = cv2_img
            print('\r', idx + 1, len(os.listdir(record_dir)), end=' ' * 10)

        out.release()
            # cv2.imwrite(output_path_layout, true_top_view)
            # prev_img = cv2_img
    print("\n-> Done!")


if __name__ == "__main__":
    cfg_path = '/CV/users/zhaohaimei3/DepthLayout_kitti_odometry_loss_transformerright_object_argoverse_A100-1024/config/cfg_kitti_baseline_odometry_boundary_ce_iou_1024_20_B1.py'# path to cfg file
    # model_path = '/public/data1/users/zhaohaimei3/DepthLayout_kitti_odometry/log/depthLayoutKittiOdom_4gpus/epoch_75.pth'# path to model weight
    # model_path = '/CV/zhaohaimei3/DepthLayout_kitti_odometry_loss_transformerright_object_argoverse_A100/log/argo_B16/epoch_5.pth'# path to model weight
    model_path = '/CV/users/zhaohaimei3/DepthLayout_kitti_odometry_loss_transformerright_object_argoverse_A100-1024/log/8gpusB24eigen/epoch_47.pth'# path to model weight
    #model_path = '/CV/users/zhaohaimei3/DepthLayout_kitti_odometry_loss_transformerright_object_argoverse_A100-1024/log/epoch_75.pth'  # path to model weight
    model_path_object = '/CV/users/zhaohaimei3/DepthLayout_kitti_odometry_loss_transformerright_object_focalloss_1024/log/object_boundaryloss_mul20_iou_mul20_ce/kitti_object.pth'# path to model weight
    img_path = '/CV/datasets/kitti_data/odometry/dataset/sequences/07/image_2/'
    output_path = '/CV/users/zhaohaimei3/DepthLayout_kitti_odometry_loss_transformerright_object_argoverse_A100-1024/pred_layout/' # dir for saving depth maps
    output_path2 = '/CV/users/zhaohaimei3/DepthLayout_kitti_odometry_loss_transformerright_object_argoverse_A100-1024/pred_disps_mono/' # dir for saving depth maps
    evaluate(cfg_path, model_path, img_path, output_path, output_path2)
