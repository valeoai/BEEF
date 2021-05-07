import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import average_precision_score
from bootstrap.lib.logger import Logger


class mAP(nn.Module):
    def __init__(self, engine=None, mode=None):
        super(mAP, self).__init__()
        self.engine = engine
        self.mode = mode
        self.dataset = self.engine.dataset[self.mode]
        self.n_classes = len(self.dataset.classid_to_ix)
        self.all_gts, self.all_preds = [], []

        self.seen_idx = set()

        if self.engine is not None:
            self.engine.register_hook(
                '%s_on_end_epoch'%mode,
                self.end_epoch)

    def end_epoch(self):
        self.all_gts = np.concatenate(self.all_gts)
        self.all_preds = np.concatenate(self.all_preds)
        aps = []
        for classid, event_id in enumerate(self.dataset.classid_to_ix):
            if event_id != -1:
                ap = average_precision_score(self.all_gts[:,classid], self.all_preds[:,classid])*100
                event_name = self.dataset.ix_to_event[event_id]
                Logger().log_value('%s_epoch.%s_ap' % (self.mode, event_name), ap, should_print=True)
                aps.append(ap)
        mean_ap = np.mean(aps)

        Logger().log_value('%s_epoch.mAP' % self.mode, mean_ap, should_print=True)

        self.all_gts, self.all_preds = [], []
        self.seen_idx = set()
        return aps

    def forward(self, cri_out, net_out, batch):
        # n_frames x n_classes
        unseen_items = torch.tensor([idx not in self.seen_idx for idx in batch['idx']])

        y_true = batch['y_true'][unseen_items].view(-1)
        n_frames = len(y_true)
        y_true_cat = torch.zeros(n_frames, self.n_classes)
        y_true_cat[torch.arange(n_frames), y_true] = 1
        try:
            y_pred = net_out['y_pred'][unseen_items].view(n_frames, -1)
        except:
            return {}
        y_pred = torch.softmax(y_pred, -1)

        self.all_gts.append(y_true_cat.cpu().detach().numpy())
        self.all_preds.append(y_pred.cpu().detach().numpy())
        self.seen_idx.update(batch['idx'])
        return {}
