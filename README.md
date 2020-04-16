# EfficientDet-Pytorch
This project is a kind of implementation of EfficientDet using mmdetection.

It is based on the

* the paper [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)
* [official TensorFlow implementation](https://github.com/google/automl)
* [Pytorch implementation of EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch)

## Models

| Variant | mAP(val2017) | Params | FLOPs | mAP(val2017) in paper | Params in paper | FLOPs in paper |
| ------- | ------------ | ------ | ----- | --------------------- | --------------- | -------------- |
| D0      | 31.6         | 4.1M   | 2.7B  | 33.5                  | 3.9M            | 2.5B           |
| D1      |              |        |       | 39.1                  | 6.6M            | 6.1B           |
| D2      |              |        |       | 42.5                  | 8.1M            | 11B            |
| D3      |              |        |       | 45.9                  | 12M             | 25B            |
| D4      |              |        |       | 49.0                  | 21M             | 55B            |
| D5      |              |        |       | 50.5                  | 34M             | 135B           |
| D6      |              |        |       | 51.3                  | 52M             | 226B           |

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
        > 	  ├── mmdet
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

- [2020-04-17] add efficientdet-d0 training config
- [2020-04-16] add efficientnet.py and retina_sepconv_head.py
- [2020-04-06] create this repository.

## Notice

1. The number of bifpn in the latest version of paper is a little different from the first version, but the parameters and flops are the same. I use the structure in the latest version of paper.
2. Training from scratch is a time-consuming task. For exmaple, it took me 4 days to train D0 from scratch using 4 GTX TiTAN V GPUs.



