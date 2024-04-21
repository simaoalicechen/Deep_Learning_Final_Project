def compute_mean_std(data_loader):
    """Compute mean and standard deviation for a given dataset"""

    means, stds = torch.zeros(3), torch.zeros(3)
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.squeeze(0)
        means += torch.Tensor([torch.mean(x[i]) for i in range(3)])
        stds += torch.Tensor([torch.std(x[i]) for i in range(3)])
        if batch_idx % 1000 == 0 and batch_idx:
            print('{:d} images processed'.format(batch_idx))

    mean = torch.div(means, len(data_loader.dataset))
    std = torch.div(stds, len(data_loader.dataset))
    print('Mean = {}\nStd = {}'.format(mean.tolist(), std.tolist()))
    return mean, std


def load_dataset(root_dir, batch_size):
    """Load data from image folder"""
    # mean, std = [0.5066, 0.4261, 0.3836], [0.2589, 0.2380, 0.2340]
    mean, std = [0.5] * 3, [0.5] * 3
    normalize = transforms.Normalize(mean=mean, std=std)
    train_data = ImageFolder(root=root_dir,
            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return data_loader