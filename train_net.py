import argparse
import logging
import os
from datetime import datetime
from torch.autograd import Variable
from data import get_loader
from models.My.Mynet import *
from utils import clip_gradient, adjust_lr
from lr_scheduler import LR_Scheduler

saveLog = 'logs'
os.makedirs(saveLog, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
fileName = datetime.now().strftime('day' + '%Y-%m-%d-%H')
handler = logging.FileHandler(os.path.join(saveLog, fileName + '.log'))
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--numworker', type=int, default=0, help='num_workers')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
opt = parser.parse_args()

print('Learning Rate: {}'.format(opt.lr))
# build models
model = Net()
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

image_root = r'E:\data\ORSSD\train\image/'
gt_root = r'E:\data\ORSSD\train\GT/'

train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize,
                          trainsize=opt.trainsize, num_workers=opt.numwoker)

total_step = len(train_loader)
# 余弦退火
scheduler = LR_Scheduler('cos', opt.lr, opt.epoch, total_step, warmup_epochs=5)

def joint_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


t_loss = []


def train(train_loader, model, optimizer, epoch):
    perLoss = 0.
    model.train()
    for i, pack in enumerate(train_loader, start=1):

        optimizer.zero_grad()

        scheduler(optimizer, i, epoch)

        images, gts = pack
        images = Variable(images)
        gts = Variable(gts)

        images = images.cuda()
        gts = gts.cuda()

        refine, fk, d1_out, d2_out, d3_out, d4_out = model(images)

        loss1 = joint_loss(refine, gts)
        loss2 = joint_loss(fk, gts)
        loss3 = joint_loss(d1_out, gts)
        loss4 = joint_loss(d2_out, gts)
        loss5 = joint_loss(d3_out, gts)
        loss6 = joint_loss(d4_out, gts)

        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        perLoss += loss.data.cpu().numpy()
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 20 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {:.6f}, Loss: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           optimizer.param_groups[0]['lr'], loss.data))
            logger.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {:.6f}, Loss: {:.4f}'.
                        format(datetime.now(), epoch, opt.epoch, i, total_step,
                               optimizer.param_groups[0]['lr'], loss.data))
    t_los_mean = perLoss / total_step
    print('Train Epoch: {}, m_Loss: {:.4f}'.format(epoch, t_los_mean))

    t_loss.append(t_los_mean)
    save_path = r'./model/my_net/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch == opt.epoch:
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    with open("train_loss.txt", 'w') as h:
        for p in range(len(t_loss)):
            c = str(t_loss[p])
            h.write(c + '\n')


if __name__ == '__main__':
    print("START!")
    for epoch in range(1, opt.epoch + 1):
        # adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
