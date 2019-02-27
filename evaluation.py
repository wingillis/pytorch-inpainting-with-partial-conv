import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image

def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor([1, 1, 1]) + torch.Tensor([1, 1, 1])
    x = x.transpose(1, 3)
    return x

def evaluate(model, dataset, device, filename):
    image, mask, gt = zip(*[dataset[i] for i in range(8)])
    image = torch.stack(image)
    mask = torch.stack(mask)
    gt = torch.stack(gt)
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    # output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output
    image = image.cpu()
    mask = mask.cpu()
    output = output.cpu()
    output_comp = output_comp.cpu()
    gt = gt.cpu()

    grid = make_grid(
        torch.cat((unnormalize(image), mask, unnormalize(output),
                   unnormalize(output_comp), unnormalize(gt)), dim=0))
    save_image(grid, filename)
