##  UACENet: Uncertain area attention and cross-image context extraction network for polyp segmentation

### Paper Information
```
Wang Z, Gao F, Yu L, et al. UACENet: Uncertain area attention and cross‐image context extraction network for polyp segmentation[J]. International Journal of Imaging Systems and Technology.

https://doi.org/10.1002/ima.22906
```


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
```

### 2. Training

```bash
train.py 
```

###  3. Testing

```bash
test.py


