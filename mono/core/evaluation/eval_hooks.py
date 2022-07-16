import os
import os.path as osp
import cv2
import matplotlib.pyplot as plt

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import Hook
from mmcv.parallel import scatter, collate
from torch.utils.data import Dataset
from .pixel_error import *

MIN_DEPTH = 1e-3
MAX_DEPTH = 80

def change_input_variable(data):
    for k, v in data.items():
        data[k] = torch.as_tensor(v).float()
    return data

def unsqueeze_input_variable(data):
    for k, v in data.items():
        data[k] = torch.unsqueeze(v, dim=0)
    return data

class NonDistEvalHook(Hook):
    def __init__(self, dataset, cfg):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.interval = cfg.get('interval', 1)
        self.out_path = cfg.get('work_dir', './')
        self.cfg = cfg

    def after_train_epoch(self, runner):
        print('evaluation..............................................')

        abs_rel = AverageMeter()
        sq_rel = AverageMeter()
        rmse = AverageMeter()
        rmse_log = AverageMeter()
        a1 = AverageMeter()
        a2 = AverageMeter()
        a3 = AverageMeter()

        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()

        for idx in range(self.dataset.__len__()):
            data = self.dataset[idx]
            data = change_input_variable(data)
            data = unsqueeze_input_variable(data)
            with torch.no_grad():
                result = runner.model(data)

            disp = result[("disp", 0, 0)]
            pred_disp, _ = disp_to_depth(disp)
            pred_disp = pred_disp.cpu()[0, 0].numpy()

            gt_depth = data['gt_depth'].cpu()[0].numpy()
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            ratio = np.median(gt_depth) / np.median(pred_depth)
            pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            abs_rel_, sq_rel_, rmse_, rmse_log_, a1_, a2_, a3_ = compute_errors(gt_depth, pred_depth)

            abs_rel.update(abs_rel_)
            sq_rel.update(sq_rel_)
            rmse.update(rmse_)
            rmse_log.update(rmse_log_)
            a1.update(a1_)
            a2.update(a2_)
            a3.update(a3_)
            print('a1_ is ', a1_)

        print('a1 is ', a1.avg)


class DistEvalHook(Hook):
    def __init__(self, dataset, interval=1, cfg=None):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.interval = interval
        self.cfg = cfg
        self.count = 0
    def eval_layout(self, runner):
        pred = np.squeeze(
            torch.argmax(
                result["topview"].detach(),
                1).cpu().numpy())
        true = np.squeeze(inputs["both"].detach().cpu().numpy())
        # print("true",true.shape)
        # print("pred",pred.shape)
        iou += mean_IU(pred, true)
        mAP += mean_precision(pred, true)

        return None
    def after_train_epoch(self, runner):
        self.count = self.count + 1
        print('evaluation..............................................'+str(self.count))

        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = [None for _ in range(len(self.dataset))]
        if runner.rank == 0:
            prog_bar = mmcv.ProgressBar(len(self.dataset))

        t = 0
        for idx in range(runner.rank, len(self.dataset), runner.world_size):
        # for idx in range(len(self.dataset)):
        #     print("len of dataset***********",len(self.dataset))
            iou_list, mAP_list = np.array([0., 0.]), np.array([0., 0.])
            iou_listB, mAP_listB = np.array([0., 0.]), np.array([0., 0.])
            data = self.dataset[idx]
            data = change_input_variable(data)
            # print("data", data["color_aug", 0, 0].shape)

            data_gpu = scatter(collate([data], samples_per_gpu=1), [torch.cuda.current_device()])[0]
            # print("data_gpu",data_gpu["color_aug", 0, 0].shape)

            # compute output
            with torch.no_grad():
                t1 = cv2.getTickCount()
                result = runner.model(data_gpu)
                t2 = cv2.getTickCount()
                t += cv2.getTickFrequency() / (t2-t1)

            # if data['gt_depth'] is not None:
            if 'gt_depth' in data.keys():
                disp = result[("disp", 0, 0)]

                pred_disp, _ = disp_to_depth(disp)
                pred_disp = pred_disp.cpu()[0, 0].numpy()

                gt_depth = data['gt_depth'].cpu().numpy()
                gt_height, gt_width = gt_depth.shape[:2]

                pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
                pred_depth = 1 / pred_disp

                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                 0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

                pred_depth = pred_depth[mask]
                gt_depth = gt_depth[mask]

                ratio = np.median(gt_depth) / np.median(pred_depth)
                if self.cfg.data['stereo_scale']:
                    pred_depth *= 36
                else:
                    pred_depth *= ratio

                pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
                pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

                abs_rel_, sq_rel_, rmse_, rmse_log_, a1_, a2_, a3_ = compute_errors(gt_depth, pred_depth)
            # if runner.rank == 0:
            #     if idx % 5 == 0:
            #         img_path = os.path.join(self.cfg.work_dir, 'visual_{:0>4d}.png'.format(idx))
            #         vmax = np.percentile(pred_disp, 95)
            #         plt.imsave(img_path, pred_disp, cmap='magma', vmax=vmax)
            pred = np.squeeze(
                torch.argmax(
                    result["topview"].detach(),
                    1).cpu().numpy())
            true = np.squeeze(data["bothS",0,0].detach().cpu().numpy())
            # print("true",true.shape)
            # print("pred",pred.shape)
            iou_list += mean_IU(pred, true)
            # iou_ = np.mean(iou_list)
            mAP_list += mean_precision(pred, true)
            predB = np.squeeze(
                torch.argmax(
                    result["topviewB"].detach(),
                    1).cpu().numpy())
            trueB = np.squeeze(data["bothD", 0, 0].detach().cpu().numpy())
            # print("true",true.shape)
            # print("pred",pred.shape)
            iou_listB += mean_IU(predB, trueB)
            # iou_ = np.mean(iou_list)
            mAP_listB += mean_precision(predB, trueB)
            # mAP_ = np.mean(mAP_list)
            # print("len of ioulist",len(iou_list))
            # print("len of mAP_list",len(mAP_list))
            result = {}
            if 'gt_depth' in data.keys():
            # if data['gt_depth'] is not None:
                result['abs_rel'] = abs_rel_
                result['sq_rel'] = sq_rel_
                result['rmse'] = rmse_
                result['rmse_log'] = rmse_log_
                result['a1'] = a1_
                result['a2'] = a2_
                result['a3'] = a3_
                result['scale'] = ratio
                # result['iou_mean'] = iou_
                # result['mAP_mean'] = mAP_
            else:
                result['abs_rel'] = 0.
                result['sq_rel'] = 0.
                result['rmse'] = 0.
                result['rmse_log'] = 0.
                result['a1'] = 0.
                result['a2'] = 0.
                result['a3'] = 0.
                result['scale'] = 0.
                # result['iou_mean'] = 0.
                # result['mAP_mean'] = 0.
            result['iou_road'] = iou_list[1]
            result['mAP_road'] = mAP_list[1]
            result['iou_vehicle'] = iou_listB[1]
            result['mAP_vehicle'] = mAP_listB[1]
            results[idx] = result

            batch_size = runner.world_size
            if runner.rank == 0:
                for _ in range(batch_size):
                    prog_bar.update()


        if runner.rank == 0:
            print('\n')
            print('FPS:', t/len(self.dataset))

            print('\n')
            dist.barrier()
            for i in range(1, runner.world_size):
                tmp_file = osp.join(runner.work_dir, 'temp_{}.pkl'.format(i))
                tmp_results = mmcv.load(tmp_file)
                for idx in range(i, len(results), runner.world_size):
                    results[idx] = tmp_results[idx]
                os.remove(tmp_file)
            self.evaluate(runner, results)
        else:
            tmp_file = osp.join(runner.work_dir,
                                'temp_{}.pkl'.format(runner.rank))
            mmcv.dump(results, tmp_file)
            dist.barrier()
        dist.barrier()

    def evaluate(self, runner, results):
        raise NotImplementedError


class DistEvalMonoHook(DistEvalHook):
    def evaluate(self, runner, results):
        if mmcv.is_str(results):
            assert results.endswith('.pkl')
            results = mmcv.load(results)
        elif not isinstance(results, list):
            raise TypeError('results must be a list of numpy arrays or a filename, not {}'.format(type(results)))

        abs_rel = AverageMeter()
        sq_rel = AverageMeter()
        rmse = AverageMeter()
        rmse_log = AverageMeter()
        a1 = AverageMeter()
        a2 = AverageMeter()
        a3 = AverageMeter()
        # iou_mean = AverageMeter()
        # mAP_mean = AverageMeter()
        iou_vehicle = AverageMeter()
        mAP_vehicle = AverageMeter()
        iou_road = AverageMeter()
        mAP_road = AverageMeter()
        scale = AverageMeter()

        print('results len is ', results.__len__())
        ratio = []
        for result in results:
            abs_rel.update(result['abs_rel'])
            sq_rel.update(result['sq_rel'])
            rmse.update(result['rmse'])
            rmse_log.update(result['rmse_log'])
            a1.update(result['a1'])
            a2.update(result['a2'])
            a3.update(result['a3'])
            scale.update(result['scale'])
            # iou_mean.update(result['iou_mean'])
            # mAP_mean.update(result['mAP_mean'])
            iou_vehicle.update(result['iou_vehicle'])
            mAP_vehicle.update(result['mAP_vehicle'])
            iou_road.update(result['iou_road'])
            mAP_road.update(result['mAP_road'])
            # ratio.append(result['scale'])

        runner.log_buffer.output['abs_rel'] = abs_rel.avg
        runner.log_buffer.output['sq_rel'] = sq_rel.avg
        runner.log_buffer.output['rmse'] = rmse.avg
        runner.log_buffer.output['rmse_log'] = rmse_log.avg
        runner.log_buffer.output['a1'] = a1.avg
        runner.log_buffer.output['a2'] = a2.avg
        runner.log_buffer.output['a3'] = a3.avg
        runner.log_buffer.output['scale mean'] = scale.avg
        # runner.log_buffer.output['scale std'] = np.std(ratio)
        # runner.log_buffer.output['iou_mean'] = iou_mean.avg
        # runner.log_buffer.output['mAP_mean'] = mAP_mean.avg
        runner.log_buffer.output['iou_vehicle'] = iou_vehicle.avg
        runner.log_buffer.output['mAP_vehicle'] = mAP_vehicle.avg
        runner.log_buffer.output['iou_road'] = iou_road.avg
        runner.log_buffer.output['mAP_road'] = mAP_road.avg
        runner.log_buffer.ready = True
