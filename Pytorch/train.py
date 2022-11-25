import torch
from torch.utils.tensorboard import SummaryWriter
from callback import EarlyStopping, Model_checkpoint, CSV_log

classes = ('angry', 'disgust', 'fear', 'happy',
           'neutral', 'sad', 'surprise')


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
        for inputs, labels in ds_train:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            # with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), labels.long())
            loss.backward()
            optimizer.step()
            num_image += inputs.size(0)
            loss_ += loss.item() * num_image
            _, num = torch.max(outputs, 1)
            train_acc += torch.sum(num == labels)

        total_loss_train = loss_ / num_image
        total_acc_train = (train_acc / num_image).item()

        return model, total_loss_train, total_acc_train

    def valid(self, ds_valid, model, criterion, device):
        model.eval()
        loss_ = 0
        valid_acc = 0
        correct_count = 0
        num_image = 0
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}
        for inputs, labels in ds_valid:
            inputs = inputs.to(device)
            labels = labels.to(device)
            num_image += inputs.size(0)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), labels.long())
            loss_ += loss.item() * num_image
            Max, num = torch.max(outputs, 1)
            valid_acc += torch.sum(num == labels)
            _, preds = torch.max(outputs.data, 1)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
            correct_count += (preds == labels).sum().item()
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

        total_loss_valid = loss_ / num_image
        total_acc_valid = (valid_acc / num_image).item()
        return model, total_loss_valid, total_acc_valid


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
            model, total_loss_valid, total_acc_valid = valid(ds_valid, model, criterion, device)
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
