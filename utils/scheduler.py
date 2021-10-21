from torch.optim.lr_scheduler import _LRScheduler

class MultiConstantLR(_LRScheduler):

    def __init__(self, optimizer, milestones, last_epoch=-1, verbose=False):
        
        self.milestones = milestones
        super(MultiConstantLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):

        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]

        return [self.milestones[self.last_epoch] for group in self.optimizer.param_groups]