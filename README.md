##  UACENet: Uncertain area attention and cross-image context extraction network for polyp segmentation

### Requirements

* torch
* torchvision
* scipy
* PIL
* numpy
* tqdm

### Data
```
$ data
train
├── images
├── masks
valid
├── images
├── masks
test
├── images
├── masks
```

### 1. set up parameters
```bash
opt.py

### 2. Training

```bash
python train.py 
```

###  3. Testing

```bash
python test.py


