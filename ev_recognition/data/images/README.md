# Image Dataset Folder

Create one folder per class:

```text
ambulance/
fire_truck/
police/
normal/
```

Put `.jpg` or `.png` images inside each class folder before running:

```bash
python -m evr.train_vision --data data/images --epochs 12
```
