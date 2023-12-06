import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging
import random
from utils import DiceLoss
from tensorboardX import SummaryWriter
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from torch.nn.modules.loss import CrossEntropyLoss

def trainer_synapse(args, model, snapshot_path, average_loss=0.0, average_dice=0.0):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of the train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    best_val_loss = 100 # Initialize with a very high value
    best_model_path = None
    a = 0
    for epoch_num in iterator:
        total_loss_epoch = 0.0
        total_dice_epoch = 0.0

        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.1 * loss_ce + 0.9 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            total_loss_epoch += loss.item()
            total_dice_epoch += loss_ce.item()

        # Calculate average loss and Dice score for the epoch
        average_loss_epoch = total_loss_epoch / len(trainloader)
        average_dice_epoch = total_dice_epoch / len(trainloader)

        # Log the average values for this epoch
        logging.info(
            'Epoch %d : average loss : %f, average dice: %f' % (epoch_num, average_loss_epoch, average_dice_epoch))

        # Update the global average loss and Dice score
        average_loss = (average_loss * epoch_num + average_loss_epoch) / (epoch_num + 1)
        average_dice = (average_dice * epoch_num + average_dice_epoch) / (epoch_num + 1)

        save_interval = 50
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

            # Check if this model is the best based on validation loss
            # You need to load the validation data and calculate the validation loss here
            validation_loss = loss  # Implement this function
            if validation_loss < best_val_loss:
                a += 1
                best_val_loss = validation_loss
                best_model_path = save_mode_path

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
        logging.info("Loaded the best model with validation loss: {}".format(best_val_loss))
        save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
        torch.save(model.state_dict(), save_mode_path)
        print(f"Model best model saved to: {save_mode_path}")
    logging.info("Final average loss: %f, Final average dice: %f" % (average_loss, average_dice))
    writer.close()
    return "Training Finished!"
