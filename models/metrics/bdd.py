import dcor
import numpy as np
import torch
import torch.nn as nn

from bootstrap.lib.logger import Logger
from scipy.spatial.distance import correlation


class BDDDrive(nn.Module):
    def __init__(self, engine, mode):
        super(BDDDrive, self).__init__()
        self.engine = engine
        self.mode = mode
        self.accel_mae = nn.L1Loss()
        self.course_mae = nn.L1Loss()
        self.all_gts = {'course': [], 'accel': []}
        self.all_preds = {'course': [], 'accel': []}
        dataset = self.engine.dataset[self.mode]
        self.course_stat = dataset.course_stat
        self.accel_stat = dataset.accel_stat

        if self.engine is not None:
            self.engine.register_hook(
                '%s_on_end_epoch'%mode,
                self.end_epoch)

    def end_epoch(self):
        self.all_gts['course'] = np.concatenate(self.all_gts['course'])
        self.all_preds['course'] = np.concatenate(self.all_preds['course'])
        self.all_gts['accel'] = np.concatenate(self.all_gts['accel'])
        self.all_preds['accel'] = np.concatenate(self.all_preds['accel'])
        course_correl = dcor.distance_correlation(self.all_gts['course'], self.all_preds['course'])
        accel_correl = dcor.distance_correlation(self.all_gts['accel'], self.all_preds['accel'])
        Logger().log_value('%s_epoch.course_correl' % self.mode, course_correl, should_print=True)
        Logger().log_value('%s_epoch.accel_correl' % self.mode, accel_correl, should_print=True)
        self.all_gts = {'course': [], 'accel': []}
        self.all_preds = {'course': [], 'accel': []}
        return None


    def forward(self, cri_out, net_out, batch):
        preds = net_out['preds']
        preds_course = preds[:,0]
        preds_accel = preds[:,1]

        if cri_out['normalize_outputs']:
            preds_course = preds_course*self.course_stat['std'] + self.course_stat['mean']
            preds_accel = preds_accel*self.accel_stat['std'] + self.accel_stat['mean']

        gt_course = batch['course_value'][:,0]
        gt_accel = batch['accelerator_value'][:,0]
        self.all_preds['course'].append(preds_course.cpu().detach().numpy())
        self.all_preds['accel'].append(preds_accel.cpu().detach().numpy())


        self.all_gts['course'].append(gt_course.cpu().detach().numpy())
        self.all_gts['accel'].append(gt_accel.cpu().detach().numpy())

        accel_mae = self.accel_mae(preds_accel, gt_accel)
        course_mae = self.course_mae(preds_course, gt_course)
        return {'accel_mae': accel_mae,
                'course_mae': course_mae}
