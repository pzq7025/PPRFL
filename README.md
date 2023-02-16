# Privacy-Preserving Roub

This is the code of experiment in paper "Fayao Wang, Yuanyuan He, Peizhi Li, Xinyu Wei, Yunchuan Guo. Privacy-Preserving Robust Federated Learning With Distributed Differential Privacy. 2022 IEEE International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom)"

## Requirement

    pip install -r requirements.txt

## Instruction

### Parameter Description

    python3 main_fed.py --help

### Example

    python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0

    python main_fed.py --dataset cifar --iid --num_channels 1 --model cnn --epochs 50 --gpu 0

### Noise Parameter

    models/Update.py: noise_multiplier



