import torch
from torch.utils.tensorboard import SummaryWriter
from callback import EarlyStopping, Model_checkpoint, CSV_log
from tqdm import tqdm


class Trainer:
    def __init__(self, logdir="./logs", csv_log_dir="./csv_log", model_checkpoint_dir="./model"):
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        self.csv_log_dir = csv_log_dir
        self.model_checkpoint_dir = model_checkpoint_dir

    def train(self, ds_train, model, criterion, optimizer, device):
        model.train()
        loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0

        for i, data in tqdm(enumerate(ds_train), total=len(ds_train), leave=False):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Train with FP16
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            optimizer.step()
            loss.backward()
            optimizer.zero_grad()
            loss_tr += loss.item()
            _, preds = torch.max(outputs.data, dim=1)
            correct_count += (preds == torch.max(labels, dim=1)[1]).sum().item()
            n_samples += labels.size(0)

        acc = 100 * correct_count / n_samples
        loss = loss_tr / n_samples

        return model, loss, acc

    def valid(self, ds_valid, model, criterion, device):
        net = model.eval()
        with torch.no_grad():
            loss_, correct_count, n_samples = 0.0, 0.0, 0.0
            for data in tqdm(ds_valid, total=len(ds_valid), leave=False):
                inputs, labels = data
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device,
                                                                                 non_blocking=True)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss_ += loss.item()
                _, preds = torch.max(outputs.data, 1)
                correct_count += (preds == torch.max(labels, dim=1)[1]).sum().item()
                n_samples += labels.size(0)

        acc = 100 * correct_count / n_samples
        loss = loss_ / n_samples
        return model, loss, acc

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
            exp_lr.step()
            metrics = {'train_loss': train_losses, 'train_acc': train_accs, 'val_loss': valid_losses,
                       'val_acc': valid_accs}

            Model_checkpoint(path=self.model_checkpoint_dir, metrics=metrics, model=model,
                             monitor='val_acc', verbose=True,
                             file_name="best.pth")
            if early_stopping.Early_Stopping(monitor='val_acc', metrics=metrics, patience=30, verbose=True):
                break
            print("Epoch:", epoch + 1, "- Train Loss:", total_loss_train, "- Train Accuracy:", total_acc_train,
                  "- Validation Loss:", total_loss_valid, "- Validation Accuracy:", total_acc_valid, "- LR:",
                  optimizer.param_groups[0]['lr'])
        return model, optimizer, train_losses, valid_losses
