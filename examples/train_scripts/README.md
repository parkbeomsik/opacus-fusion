# Training scripts

Resnet18 with 59.6% (2012 sec)
```bash
python train_cifar10.py --dpsgd_mode elegant --batch_size 4096 --gn-groups 16 --wd 5e-5 --momentum 0.9
```

Resnet18 with 58.4% (2857 sec)
```bash
python train_cifar10.py --dpsgd_mode naive --batch_size 4096 --physical_batch_size 512 --gn-groups 16 --wd 5e-5 --momentum 0.9
```

Resnet18 with 59.6% (2012 sec)
```bash
python train_cifar10.py --dpsgd_mode elegant --batch_size 4096 --gn-groups 16 --wd 5e-5 --momentum 0.9
```

Resnet18 with 58.3% (3199 sec)
```bash
python train_cifar10.py --optim RMSprop --gn-groups 16 --batch_size=512 --physical_batch_size=512 --lr_schedule none --epochs 100 -c 1.2 --lr 1e-3
```

Resnet18 with 59.1% (3354 sec)
```bash
python train_cifar10.py --optim RMSprop --gn-groups 16 --batch_size=1024 --lr_schedule none --epochs 100 -c 1.2 --lr 1e-3 --dpsgd_mode reweight
```

Resnet50 with 55.7%
```bash
python train_cifar10.py --architecture resnet50 --dpsgd_mode elegant --batch_size 2048 -c 1.0 --wd 5e-4 --momentum 0.9 --epsilon 4.0
```

Resnet50 with 53.89%
```bash
python train_cifar10.py --architecture resnet50 --dpsgd_mode naive --batch_size 4096 --physical_batch_size 256  -c 1.0 --wd 5e-4 --momentum 0.9 --epsilon 4.0
```

Resnet50 with 54.4%
```bash
python train_cifar10.py --architecture resnet50 --dpsgd_mode reweight --batch_size 4096 --physical_batch_size 1024 -c 1.0 --wd 5e-4 --momentum 0.9 --epsilon 4.0
```

Resnet18 (from PLACE365) with 66.90%
```bash
python train_cifar10_fine_tuning.py --architecture resnet18 --dpsgd_mode elegant --batch_size 4096 -c 1.0 --wd 5e-4 --momentum 0.9 --epsilon 4.0 --pretrained_path data/places365_resnet18_20220314.npz
```

Resnet18 (from PLACE365) with 66.90%
```bash
python train_cifar10_fine_tuning.py --architecture resnet18 --dpsgd_mode elegant --batch_size 4096 -c 1.0 --wd 5e-4 --momentum 0.9 --epsilon 4.0 --pretrained_path data/places365_resnet18_20220314.npz
```

Resnet50 (from PLACE365) with 66.90%
```bash
python train_cifar10_fine_tuning.py --architecture resnet50 --dpsgd_mode elegant --batch_size 4096 -c 1.0 --wd 5e-4 --momentum 0.9 --epsilon 4.0 --pretrained_path data/places365_resnet50_20220314.npz
```

Resnet152 (from PLACE365) with 
```bash
python train_cifar10_fine_tuning.py --architecture resnet152 --gn-groups 16 --dpsgd_mode reweight --batch_size 2048 --physical_batch_size 512 -c 1.0 --wd 5e-4 --momentum 0.9 --epsilon 4.0 --optim RMSprop --lr 1e-3 --pretrained_path data/places365_resnet152_20220314.npz
```