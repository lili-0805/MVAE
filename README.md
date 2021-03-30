# MVAE -multichannel variational autoencoder for audio source separation-

This repository provides official PyTorch implementation of multichannel variational autoencoder (MVAE) proposed in the following papers. 
We also provide pretrained models for speaker-closed situation described in the paper #1 and speaker-open situation described in the paper #2. 

FastMVAE is coming soon...

1. Hirokazu Kameoka, Li Li, Shota Inoue, and Shoji Makino, "Supervised Determined Source Separation with Multichannel Variational Autoencoder," Neural Computation, vol. 31, no. 9, pp. 1891-1914, Sep. 2019.
2. Li Li, Hirokazu Kameoka, Shota Inoue, and Shoji Makino, "FastMVAE: A fast optimization algorithm for the multichannel variational autoencoder method," IEEE Accesss, vol. 8, pp. 228740-228753, Dec. 2020.


## Dependencies

Code was tested using following packages.

* Python 3.7.0
* PyTorch 1.6.0
* Scipy 1.4.1
* Numpy 1.17.0

## Download
Get MVAE code

```bash
$ git clone https://github.com/lili-0805/MVAE.git
```

Using download script to download training dataset, test dataset, and pretrained models.
The test samples were generated using the VCC dataset. Namely, the test samples are speaker-closed for models trained using the VCC dataset, and speaker-open for models trained using the WSJ0 dataset.
Because of the license of WSJ0 database, we do not provide training dataset of WSJ0. Please download WSJ0 database and prepare trainging dataset described in our paper #2 by yourselves. 

```bash
$ cd MVAE
$ bash download.sh dataset-VCC
$ bash download.sh test-samples
$ bash download.sh model-VCC
$ bash download.sh model-WSJ0
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

To test MVAE algorithm with pretrained VCC model, run the script below. This command initializes MVAE with ILRMA.

```python
$ python test_mvae.py --input_root ./data/test_input/ --output_root ./output/  
  --n_itr0 30 --n_itr1 30 --model_path ./pretrained_model/model-vcc/1000.model --gpu 0
```

To test MVAE algorithm with pretrained WSJ0 model, run the script below. This command initializes MVAE with identity matrix (i.e. w/o initialization algorithm).

```python
$ python test_mvae.py --input_root ./data/test_input/ --output_root ./output/  
  --n_itr0 0 --n_itr1 60 --n_src 101 --model_path ./pretrained_model/model-wsj/1000.model --gpu 0
```

## Update history

* Release pretrained speaker-independent model. (March 30, 2021)
* Update MVAE implementation by adding initialization process of latent variable "z" and class label "l". (March 30, 2021)
* Release MVAE scripts and speaker-dependent training dataset and pretrained model. (May 23, 2020)

## License and ackwonledgements
License: [MIT](https://choosealicense.com/licenses/mit/)

If you find this work is useful for your research or project, please cite out papers:

* Hirokazu Kameoka, Li Li, Shota Inoue, and Shoji Makino, "Supervised Determined Source Separation with Multichannel Variational Autoencoder," Neural Computation, vol. 31, no. 9, pp. 1891-1914, Sep. 2019.
* Li Li, Hirokazu Kameoka, Shota Inoue, and Shoji Makino, "FastMVAE: A fast optimization algorithm for the multichannel variational autoencoder method," IEEE Accesss, vol. 8, pp. 228740-228753, Dec. 2020.


## See also

* Demo: http://www.kecl.ntt.co.jp/people/kameoka.hirokazu/Demos/mvae-ass/index.html
* Related work:

1. Underdetermined source separation:
Shogo Seki, Hirokazu Kameoka, Li Li, Tomoki Toda, and Kazuya Takeda, "Underdetermined Source Separation Based on Generalized Multichannel Variational Autoencoder," IEEE Access, vol. 7, No. 1, pp. 168104-168115, Nov. 2019.

2. Determined source separation and dereverberation:
Shota Inoue, Hirokazu Kameoka, Li Li, Shogo Seki, and Shoji Makino, "Joint separation and dereverberation of reverberant mixtures with multichannel variational autoencoder," in Proc. ICASSP2019, pp. 56-60, May 2019.
