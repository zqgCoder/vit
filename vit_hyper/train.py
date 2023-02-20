import os
import math
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model_vit import vit_base_patch16_224_in21k as create_model
from utils import train_one_epoch, evaluate
from VitSolver import HyperSolver


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    # 获取图像路径，标签
    if args.test_db == 'CASIA':
        from datasets.MNOR_C import read_split_data
    elif args.test_db == 'MSU':
        from datasets.CNOR_M import read_split_data
    elif args.test_db == 'NUAA':
        from datasets.CMOR_N import read_split_data
    elif args.test_db == 'OULU':
        from datasets.CMNR_O import read_split_data
    elif args.test_db == 'REPLAY':
        from datasets.CMNO_R import read_split_data
    else:
        assert 1 == 0
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # model = create_model(num_classes=2, has_logits=False).to(device)  # 是否有表示层

    # 移动到transformer里面
    # if args.weights != "":
    #     assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    #     weights_dict = torch.load(args.weights, map_location=device)
    #     # 删除不需要的权重
    #     del_keys = ['head.weight', 'head.bias']
    #     # del_keys = ['head.weight', 'head.bias'] if model.has_logits \
    #     #     else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    #     for k in del_keys:
    #         del weights_dict[k]
    #     print(model.load_state_dict(weights_dict, strict=False))

    # 设置是否冻结权重
    # if args.freeze_layers:
    #     for name, para in model.named_parameters():
    #         # 除head, pre_logits外，其他权重全部冻结
    #         if "head" not in name and "pre_logits" not in name:
    #             para.requires_grad_(False)
    #         else:
    #             print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]  # 需要优化的网络参数
    # print('需要优化的参数：', pg)
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # 调整学习率
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    solver_model = HyperSolver(config=args, device=device)

    bestLoss = 10.0
    bestAcc = 0.0
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = solver_model.train(data_loader=train_loader, epoch=epoch)
        # train_loss, train_acc = train_one_epoch(model=model,
        #                                         optimizer=optimizer,
        #                                         data_loader=train_loader,
        #                                         device=device,
        #                                         epoch=epoch)

        # scheduler.step()

        # validate
        val_loss, val_acc = solver_model.evaluate(data_loader=val_loader, epoch=epoch)
        # val_loss, val_acc = evaluate(model=model,
        #                              data_loader=val_loader,
        #                              device=device,
        #                              epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        # tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        print('epoch:{}, val_loss is {}, val_acc is {}'.format(epoch, val_loss, val_acc))
        if val_acc > bestAcc and val_loss < bestLoss:
            print('epoch {} is best'.format(epoch))
            bestModelPath = './weights/bestmodel-{}.pth'.format(args.test_db)
            torch.save(solver_model.model_hyper.state_dict(), bestModelPath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)  # 100
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lrv', type=float, default=0.001)
    parser.add_argument('--lrfv', type=float, default=0.01)
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10)  # hyper
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate of hyper')

    parser.add_argument('--test-db', type=str, default='REPLAY')

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="F:/db_tf")

    # 预训练权重路径，如果不想载入就设置为空字符 ./vit_base_patch16_224_in21k.pth
    parser.add_argument('--weights', type=str, default='./vit_base_patch16_224.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
