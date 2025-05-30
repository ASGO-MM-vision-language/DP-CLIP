from torchvision import transforms

def data_transform(opts):
    dataset_transforms = transforms.Compose([
        transforms.Resize((opts.max_size, opts.max_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return dataset_transforms

