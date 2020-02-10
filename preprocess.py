import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset


def get_mnist(data_dir='./data/mnist/', batch_size=128):
    train = datasets.MNIST(root=data_dir, train=True, download=True)
    test = datasets.MNIST(root=data_dir, train=False, download=True)

    x = torch.cat([train.data.float().view(-1,784)/255., test.data.float().view(-1,784)/255.], 0)
    y = torch.cat([train.targets, test.targets], 0)

    dataset = dict()
    dataset['x'] = x
    dataset['y'] = y

    dataloader = DataLoader(TensorDataset(x,y), batch_size=batch_size, 
                            shuffle=True, num_workers=0)
    return dataloader


def get_webcam(data_dir='./data/office31/webcam/images/', batch_size=128):
    data = datasets.ImageFolder(data_dir)

    x = torch.FloatTensor([data[i][1] for i in range(len(data))]))
    y = torch.FloatTensor([data[i][0] for i in range(len(data))]))

    data = CaffeTransform(x, y)

    dataloader = DataLoader(data, batch_size=batch_size, 
                          shuffle=True, num_workers=0)
    return dataloader


class CaffeTransform(torch.utils.data.Dataset):

    def __init__(self, X, y, train=True):
        super(CaffeTransform, self).__init__()
        self.X = X
        self.y = y
        self.mean_color = [104.0069879317889, 116.66876761696767, 122.6789143406786]  # BGR 
        self.train = train
        self.output_size = [227, 227]
        if self.train:
            self.horizontal_flip = True
            self.multi_scale = [256, 256]
        else:
            self.horizontal_flip = False
            self.multi_scale = [256, 256]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        img, target = self.X[idx], self.y[idx]
        # Flip image at random if flag is selected
        if self.horizontal_flip and np.random.random() < 0.5:
            img = cv2.flip(img, 1)

        if self.multi_scale is None:
            # Resize the image for output
            img = cv2.resize(img, (self.output_size[0], self.output_size[1]))
            img = img.astype(np.float32)
        elif isinstance(self.multi_scale, list):
            # Resize to random scale
            new_size = self.multi_scale[0]

            img = cv2.resize(img, (new_size, new_size))
            img = img.astype(np.float32)
            if new_size != self.output_size[0]:
                if self.train:
                    # random crop at output size
                    diff_size = new_size - self.output_size[0]
                    random_offset_x = np.random.randint(0, diff_size, 1)[0]
                    random_offset_y = np.random.randint(0, diff_size, 1)[0]
                    img = img[random_offset_x:(random_offset_x + self.output_size[0]), random_offset_y:(
                        random_offset_y + self.output_size[0])]
                else:
                    y, x, _ = img.shape
                    startx = x // 2 - self.output_size[0] // 2
                    starty = y // 2 - self.output_size[1] // 2
                    img = img[starty:starty + self.output_size[0], startx:startx + self.output_size[1]]
        img -= np.array(self.mean_color)
        img = torch.from_numpy(img)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img, target