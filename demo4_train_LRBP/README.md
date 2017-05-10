# Low-Rank-Bilinear-Pooling (LRBP)
### demo for training LRBP 

![alt text](https://raw.githubusercontent.com/aimerykong/Low-Rank-Bilinear-Pooling/master/demo4_train_LRBP/imdbFolder/CUB/exp/save4demo/CUB_VGG_16_SVM_bilinear_448_net-train.png "visualization")

This is a demo on how to train the low-rank bilinear pooling model, especially, how to initialize the dimension-reduction layer using PCA, how to initialize bisvm with random weights, and how to fine-tune the model.

Please run with caution ``main100_demo_PCA_conv53_forDimRed.m''. If your PC memory is not large enough, don't run it. Of course this file can be made better though.

I'm trying to make it self-contained. But it is quite messy now. I will re-organize the folder soon, modify the files, and add instructions including compiling matconvnet and where to put the data, etc.


If you find our model/method/dataset useful, please cite our work:

    @inproceedings{kong2017lowrankbilinear,
      title={Low-rank Bilinear Pooling for Fine-Grained Classification},
      author={Kong, Shu and Fowlkes, Charless},
      booktitle={CVPR},
      year={2017}
    }

Also achknowledge Yang Gao for his matlab code, the toolbox matconvnet to train the model, exportFig to save figures...


Shu Kong @ UCI
05/09/2017



