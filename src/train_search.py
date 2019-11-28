import os
import sys
import time
import glob
import numpy as np
import torch
from src import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from src.model_search import Network
from src.architect import Architect

from src.data_loader.data_generator import SignDataLoader, get_data_for_master_class

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

####################################################################################################
# TODO: set the right number of classes                                                            #
####################################################################################################
classes = {
    # TODO: Add signs from belgium? change the class used?
    "RedRoundSign": {
        "signs_classes": ['p1', 'p10', 'p11', 'p12', 'p19', 'p20', 'p23', 'p26', 'p27', 'p3', 'p5L', 'p6', 'p9', 'pax',
                          'pb', 'phx', 'pl100', 'pl120', 'pl20', 'pl30', 'pl40', 'pl5', 'pl50', 'pl60', 'pl70',
                          'pl80', 'pmx', 'p_prohibited_bicycle_and_pedestrian', 'p_prohibited_bus_and_truck',
                          'p_prohibited_other', 'prx', 'p_other', 'plo'],
        "merge_sign_classes": {
            "prx": ['pr10', 'pr100', 'pr20', 'pr30', 'pr40', 'pr45', 'pr50', 'pr60', 'pr70', 'pr80', 'prx'],
            "pmx": ['pm1.5', 'pm10', 'pm13', 'pm15', 'pm2', 'pm2.5', 'pm20', 'pm25', 'pm30', 'pm35', 'pm40', 'pm46',
                    'pm5', 'pm49', 'pm50', 'pm55', 'pm8'],
            "phx": ['ph', 'ph1.5', 'ph2', 'ph2.1', 'ph2.2', 'ph2.4', 'ph2.5', 'ph2.6', 'ph2.8', 'ph2.9', 'ph3.x', 'ph3',
                    'ph3.2', 'ph3.3', 'ph3.5', 'ph3.7', 'ph3.8', 'ph38', 'ph39', 'ph45', 'ph4', 'ph4.2', 'ph4.3',
                    'ph4.4', 'ph4.5', 'ph4.6', 'ph4.8', 'ph5', 'ph5.3', 'ph5.5', 'ph6'],
            "pax": ['pa10', 'pa12', 'pa13', 'pa14', 'pa8', 'pax'],
            "plo": ['pl35', 'pl25', 'pl15', 'pl10', 'pl110', 'pl65', 'pl90'],
            "p_other": ['pw2', 'pw2.5', 'pw3', 'pw3.2', 'pw3.5', 'pw4', 'pw4.2', 'pw4.5',
                        'p_prohibited_two_wheels_vehicules', 'p_prohibited_bicycle_and_pedestria',
                        'p_prohibited_bicycle_and_pedestrian_issues', 'p13', 'p15', 'p16', 'p17', 'p18', 'p2', 'p21',
                        'p22', 'p24', 'p25', 'p28', 'p4', 'p5R', 'p7L', 'p7R', 'p8', 'p15', 'pc']
        },
        "h_symmetry": [],
        "rotation_and_flips": {  # "pne": ('v', 'h', 'd'),
            # "pn": ('v', 'h', 'd'),
            # "pnl": ('d',),
            # "pc": ('v', 'h', 'd'),
            "pb": ('v', 'h', 'd'),
            "p_other": ('v', 'h', 'd'),
        }
    },
}
class_name = "RedRoundSign"
TRAFFIC_SIGNS_CLASSES = len(classes[class_name]["signs_classes"])
####################################################################################################


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, TRAFFIC_SIGNS_CLASSES, args.layers, criterion)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    ####################################################################################################
    # TODO: load traffic signs instead of CIFFAR 10                                                    #
    ####################################################################################################
    out_classes = classes[class_name]["signs_classes"]
    rotation_and_flips = classes[class_name]["rotation_and_flips"]
    h_symmetry_classes = classes[class_name]["h_symmetry"]
    try:
        merge_sign_classes = classes[class_name]["merge_sign_classes"]
    except KeyError:
        merge_sign_classes = None
    os.makedirs(class_name, exist_ok=True)
    mapping = {c: i for i, c in enumerate(out_classes)}
    mapping_id_to_name = {i: c for c, i in mapping.items()}

    x_train, y_train, x_test, y_test = get_data_for_master_class(class_name=class_name,
                                                                 mapping=mapping,
                                                                 mapping_id_to_name=mapping_id_to_name,
                                                                 rotation_and_flips=rotation_and_flips,
                                                                 data_dir=args.data,
                                                                 merge_sign_classes=merge_sign_classes,
                                                                 h_symmetry_classes=h_symmetry_classes,
                                                                 image_size=(32, 32),
                                                                 ignore_npz=False,
                                                                 out_classes=out_classes)
    x_train = np.rollaxis(x_train, 3, 1).astype(np.float32)
    x_test = np.rollaxis(x_test, 3, 1).astype(np.float32)
    x_train, y_train, x_test, y_test = [torch.from_numpy(a) for a in [x_train, y_train, x_test, y_test]]
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    # train_transform, valid_transform = utils._data_transforms_cifar10(args)
    # train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    ####################################################################################################

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(non_blocking=True)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data[0], n)
        top1.update(prec1.data[0], n)
        top5.update(prec5.data[0], n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
