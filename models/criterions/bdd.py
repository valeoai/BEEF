import torch
import torch.nn as nn


class BDDDriveLoss(nn.Module):
    def __init__(self, scales={}, normalize_outputs=False):
        super(BDDDriveLoss, self).__init__()
        self.accel_loss = nn.MSELoss()
        self.course_loss = nn.MSELoss()
        self.scales = scales
        self.normalize_outputs = normalize_outputs

    def forward(self, net_out, batch):
        preds = net_out['preds']
        preds_course = preds[:,0]
        preds_accel = preds[:,1]

        if self.normalize_outputs:
            gt_course = batch['course_value_standard']
            gt_accel = batch['accelerator_value_standard']
        else:
            gt_course = batch['course_value']
            gt_accel = batch['accelerator_value']


        accel_loss = self.accel_loss(preds_accel, gt_accel[:,0]) * self.scales.get('accelerator', 1)
        course_loss = self.course_loss(preds_course, gt_course[:,0]) * self.scales.get('course', 1)

        return {'loss': course_loss + accel_loss,
                'accel_loss': accel_loss,
                'course_loss': course_loss,
                'normalize_outputs': self.normalize_outputs}
