# MVAE -multichannel variational autoencoder for audio source separation-

This repository provides official PyTorch implementation of multichannel variational autoencoder (MVAE) proposed in the following papers:

* Hirokazu Kameoka, Li Li, Shota Inoue, and Shoji Makino, "Supervised Determined Source Separation with Multichannel Variational Autoencoder," Neural Computation, vol. 31, no. 9, pp. 1-24, Sep. 2019.


## Dependencies

Code was tested using following packages.

* Python 3.7.0
* PyTorch 1.4.0
* Scipy 1.4.1
* Numpy 1.17.0

## Download
Get MVAE code

```bash
git clone https://github.com/lili-0805/MVAE.git
```

Using download script to download training dataset, test dataset, and pretrained model.

```bash
$ cd MVAE
$ bash download.sh dataset-VCC
$ bash download.sh test-samples
$ bash download.sh model-VCC
```

## Usage

**1. Training networks**

To train CVAE on VCC dataset, run the training script below.

```python
$ python train_cvae.py --dataset vcc --save_root ./model/ --gpu 0
```

**2. Test MVAE**

To test MVAE algorithm with trained model, run the script below.

```python
$ python test_mvae.py --input_root ./data/test_input/ --output_root ./output/  
  --n_itr0 30 --n_itr1 30 --model_path ./model/vcc/1000.model --gpu 0
```

To test MVAE algorithm with pretrained model, run the script below.

```python
$ python test_mvae.py --input_root ./data/test_input/ --output_root ./output/  
  --n_itr0 30 --n_itr1 30 --model_path ./pretrained_model/model-vcc/1000.model --gpu 0
```

## Update history

* Release MVAE scripts and speaker-dependent training dataset and pretrained model. (May 23, 2020)

## License and ackwonledgements
License: [MIT](https://choosealicense.com/licenses/mit/)

If you find this work is useful for your research or project, please cite out paper:

* Hirokazu Kameoka, Li Li, Shota Inoue, and Shoji Makino, "Supervised Determined Source Separation with Multichannel Variational Autoencoder," Neural Computation, vol. 31, no. 9, pp. 1-24, Sep. 2019.


## See also

* Demo: http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/mvae-ass/index.html
* Related work:
1. Fast algorithm:
Li Li, Hirokazu Kameoka, and Shoji Makino, "Fast MVAE: Joint separation and classification of mixed sources based on multichannel variational autoencoder with auxiliary classifier," in Proc. ICASSP2019, pp. 546-550, May 2019.

2. Underdetermined source separation:
Shogo Seki, Hirokazu Kameoka, Li Li, Tomoki Toda, and Kazuya Takeda, "Underdetermined Source Separation Based on Generalized Multichannel Variational Autoencoder," IEEE Access, vol. 7, No. 1, pp. 168104-168115, Nov. 2019.

3. Determined source separation and dereverberation:
Shota Inoue, Hirokazu Kameoka, Li Li, Shogo Seki, and Shoji Makino, "Joint separation and dereverberation of reverberant mixtures with multichannel variational autoencoder," in Proc. ICASSP2019, pp. 56-60, May 2019.
