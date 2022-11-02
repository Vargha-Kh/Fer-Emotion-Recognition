import torch
from torch.utils.tensorboard import SummaryWriter
from callback import EarlyStopping, Model_checkpoint, CSV_log


class Trainer:
    def __init__(self, logdir="./logs", csv_log_dir="./csv_log", model_checkpoint_dir="./model"):
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        self.csv_log_dir = csv_log_dir
        self.model_checkpoint_dir = model_checkpoint_dir

    def train(self, ds_train, model, criterion, optimizer, device):
        model.train()
        loss_ = 0
        train_acc = 0
        num_image = 0
        for x, y_true in ds_train:
            optimizer.zero_grad()
            X = x.to(device)
            Y = y_true.to(device)
            logit = model(X)
            loss = criterion(logit, Y)
            loss_ += loss.item() * x.size(0)
            Max, num = torch.max(logit, 1)
            train_acc += torch.sum(num == Y)
            num_image += x.size(0)
            loss.backward()
            optimizer.step()
        total_loss_train = loss_ / num_image
        total_acc_train = train_acc / num_image

        return model, total_loss_train, total_acc_train.item()

    def valid(self, ds_valid, model, criterion, device):
        model.eval()
        loss_ = 0
        valid_acc = 0
        num_image = 0
        for x, y_true in ds_valid:
            X = x.to(device)
            Y = y_true.to(device)
            logit = model(X)
            loss = criterion(logit, Y)
            loss_ += loss.item() * x.size(0)
            Max, num = torch.max(logit, 1)
            valid_acc += torch.sum(num == Y)
            num_image += x.size(0)
        total_loss_valid = loss_ / num_image
        total_acc_valid = valid_acc / num_image
        return model, total_loss_valid, total_acc_valid.item()

    def training(self, model, ds_train, ds_valid, criterion, optimizer, reduce_on_plateau, exp_lr, device, epochs):
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []
        early_stopping = EarlyStopping()
        for epoch in range(epochs):
            model, total_loss_train, total_acc_train = self.train(ds_train, model, criterion, optimizer, device)
            self.writer.add_scalar("train_loss", total_loss_train, epoch)
            self.writer.add_scalar("train_accuracy", total_acc_train, epoch)
            train_losses.append(total_loss_train)
            train_accs.append(total_acc_train)
            with torch.no_grad():
                model, total_loss_valid, total_acc_valid = self.valid(ds_valid, model, criterion, device)
                valid_losses.append(total_loss_valid)
                valid_accs.append(total_acc_valid)
                self.writer.add_scalar("validation_loss", total_loss_valid, epoch)
                self.writer.add_scalar("validation_accuracy", total_acc_valid, epoch)
                self.writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

            scores = {'epoch': epoch, 'acc': total_acc_train, 'loss': total_loss_train, 'val_acc': total_acc_valid,
                      'val_loss': total_loss_valid, 'LR': optimizer.param_groups[0]['lr']}

            CSV_log(path=self.csv_log_dir, filename='log_file', score=scores)
            reduce_on_plateau.step(total_loss_valid)
#            exp_lr.step()
            metrics = {'train_loss': train_losses, 'train_acc': train_accs, 'val_loss': valid_losses,
                       'val_acc': valid_accs}

            Model_checkpoint(path=self.model_checkpoint_dir, metrics=metrics, model=model,
                             monitor='val_acc', verbose=True,
                             file_name="best.pth")
            if early_stopping.Early_Stopping(monitor='val_acc', metrics=metrics, patience=30, verbose=True):
                break
            print("Epoch:", epoch + 1, "- Train Loss:", total_loss_train, "- Train Accuracy:", total_acc_train,
                  "- Validation Loss:", total_loss_valid, "- Validation Accuracy:", total_acc_valid, "- LR:", optimizer.param_groups[0]['lr'])
        return model, optimizer, train_losses, valid_losses
