import argparse 
import torch.utils.data
from torchvision import datasets, transforms

from train import TrainerVaDE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5000,
                        help="number of iterations")
    parser.add_argument("--patience", type=int, default=50, 
                        help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument("--batch_size", type=int, default=100, 
                        help="Batch size")
    parser.add_argument('--pretrain', type=bool, default=True,
                        help='learning rate')
    parser.add_argument('--pretrained_path', type='str', default='./weights/pretrained_parameter.pth',
                        help='Output path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                             shuffle=True, num_workers=0)

    dataset = datasets.MNIST('./data', train=False, download=True,
                             transform=transforms.ToTensor())
    dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                                  shuffle=True, num_workers=0)
    
    vade = TrainerVaDE(args, device, dataloader, dataloader_test)
    if args.pretrain==True:
        vade.pretrian()
    vade.train()

