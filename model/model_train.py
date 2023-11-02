import numpy as np
from model.data_aug.Method_Wrapper import DataAugMethod


def model_train(model, criterion, optimizer, dataset, aug_config):

    model.train()

    for i, (x, y, path) in enumerate(dataset):
        # ----------------------------Compute Logits----------------------------#
        img = x.cuda()
        labels = y.cuda()

        # Perform dataaug with a probability of dataaug['r'], when dataaug is applied.
        r = np.random.rand(1)
        if (aug_config['name'] == 'none') or (r >= aug_config['r']):
            # --------------Training without Data Augmentation------------------#
            logits = model(img)

            loss = criterion(logits, labels).mean()
        else:
            # --------------Training with Data Augmentation---------------------#
            aug = DataAugMethod(aug_config, len(img))
            img = aug.augment_input(img)

            logits = model(img)

            loss = aug.augment_criterion(criterion, logits, labels)

        loss = loss.mean()
        # -----------------------Loss Backpropagation---------------------------#
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
