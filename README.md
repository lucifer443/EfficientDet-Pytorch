# EfficientDet-Pytorch
This project is a kind of implementation of EfficientDet using mmdetection.

It is based on the

* the paper [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)
* [official TensorFlow implementation](https://github.com/google/automl)
* [Pytorch implementation of EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)

## Models

| Variant | mAP(val2017) | Params | FLOPs   | mAP(val2017) in paper | Params in paper | FLOPs in paper |
| ------- | ------------ | ------ | ------- | --------------------- | --------------- | -------------- |
| D0      | 32.02        | 3.87M  | 2.55B   | 33.5                  | 3.9M            | 2.5B           |
| D1      | 37.78        | 6.62M  | 6.12B   | 39.1                  | 6.6M            | 6.1B           |
| D2      | ——           | 8.09M  | 11B     | 42.5                  | 8.1M            | 11B            |
| D3      | ——           | 12.02M | 24.88B  | 45.9                  | 12M             | 25B            |
| D4      | ——           | 20.7M  | 55.13B  | 49.0                  | 21M             | 55B            |
| D5      | ——           | 33.63M | 135.31B | 50.5                  | 34M             | 135B           |
| D6      | ——           | ——     | ——      | 51.3                  | 52M             | 226B           |

## Usage

1. **Install mmdetection**

   This implementation is based on [mmdetection](https://github.com/open-mmlab/mmdetection)(v1.1.0+8732ed9). Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.

2. **Copy the codes to mmdetection directory**

   ```shell
   cp -r mmdet/ ${MMDETECTION_PATH}/
   cp -r configs/ ${MMDETECTION_PATH}/
   ```

 3. **Prepare data**

     The directories should be arranged like this:
     
        >   mmdetection
        >     ├── mmdet
        >     ├── tools
        >     ├── configs
        >     ├── data
        >     │   ├── coco
        >     │   │   ├── annotations
        >     │   │   ├── train2017
        >     │   │   ├── val2017
        >     │   │   ├── test2017


 4. **Train D0 with 4 GPUs**

    ```shell
    CONFIG_FILE=configs/efficientdet/efficientdet_d0_4gpu.py
    ./ tools/dist_train.py ${CONFIG_FILE} 4
    ```

 5. **Calculate parameters and flops**

     ```shell
      python tools/get_flops.py ${CONFIG_FILE} --shape $SIZE $SIZE
     ```

6. **Test**

   ```shell
   python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --out  ${OUTPUT_FILE} --eval bbox
   ```

More usages can reference [mmdetection documentation](https://mmdetection.readthedocs.io/en/latest/GETTING_STARTED.html#inference-with-pretrained-models).

## Update log

- [2020-04-27] Update results and add SyncBN in backbone.
- [2020-04-20] Fix some bug in bifpn and use separate BN in head.
- [2020-04-17] Add efficientdet-d0 training config.
- [2020-04-16] Add efficientnet.py and retina_sepconv_head.py.
- [2020-04-06] Create this repository.

## Notice

1. For small reason, I can't release the model. But you can reproduce the result easily using the config file that I provide.
2.  I find the training procedure of EfficientDet is unstable and  there is a small chance that results can be 3% mAP lower.
3. The number of bifpn in the latest version of paper is a little different from the first version, but the parameters and flops are the same. I use the structure in the latest version of paper.
4. Training from scratch is a time-consuming task. For exmaple, it took me 4 days to train D0 from scratch using 4 GTX TiTAN V GPUs.



