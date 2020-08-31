from collections import OrderedDict

import torch


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter.avg) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def save_checkpoint(epoch, model, optimizer, acc, is_best, train_loss):
    print('saving checkpoint ...')
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = 'model/' + 'checkpoint_' + str(epoch) + '_' + str(train_loss) + '.tar'
    # filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'model/BEST_checkpoint_{}_{}.tar'.format(str(epoch), str(acc)))


def load_checkpoint(model, pretrained_path):
    pretrained = torch.load(pretrained_path)
    pretrained = pretrained['model']
    if type(model) == torch.nn.DataParallel:
        state_dict = pretrained.module.state_dict()
    else:
        state_dict = pretrained.state_dict()
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # print(k)
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model
