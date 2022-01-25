## ECG Classifier base on PYNQ-Z2 by using FINN Compiler

### Training
`./ecg` is modify from [this]("https://github.com/lxdv/ecg-classification").
* training setup
Put mit-bih data to folder `./ecg/mitbih`
Run `python ./scripts/annotation-generation-1d.py`
Run `python ./scripts/dataset-generation-pool.py`
Run `python train.py --config ./config/training/brevitas.json`

And get the `.pth` in the experiments folder

### Finn compile
Build finn environment on xilinx [page]("https://github.com/Xilinx/finn").

And run `finn_compile.ipynb`.

### delpoy on PYNQ
We provides a example on pynq-z2.
Upload `./onPYNQ` folder
Run `Demo-Jack.ipynb`
Run `run.ipynb`