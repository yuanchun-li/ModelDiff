# ModelDiff (pytorch)

## Scripts for different model reuse
现在每个脚本中只跑一个数据集

### Fine-tune
关于迁移学习的脚本在examples/finetune中，包括resnet18和mbnetv2两个网络。其中对于resnet18,有三个脚本conv1、layer3和fc，依次代表finetune全部网络、从中间开始以及只finetune最后的fc。对于mbnetv2，layer1、layer4和fc分别代表finetune全部网络、从中间开始以及只finetune最后的fc。

### Prune
Prune脚本在examples/prune中。这里需要使用已经训练好的student模型，而没有用teacher模型。假设使用全网络finetune的模型，那么resnet18需要使用conv1，mbnetv2需要使用layer1。在每个脚本中通过CKPT_DIR控制读取student的路径，ratio控制剪枝率。

### Distill
Distillation的脚本在examples/distill中，分为feature和output。与prune相同，distill也需要使用已经训练好的student模型，脚本中的TEACHER_DIR就是模型文件夹的路径。

## 运行环境
运行环境已经导出到environment.yml中，可以使用如下命令创建conda环境，名为modeldiff：
```
 conda env create -f environment.yml

```
注意pytorch使用的版本是1.0.0，可以需要检查一下版本是否符合。

## load model脚本
相关代码在load_model.py中，脚本在examples/load.sh中。在load_model.py主要是根据模型结构构造模型（resnet或mbnet，output classes），然后读取ckpt。

## 数据集下载
所有的数据集应该放在./data目录下

### [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
```
Flower102/
|    imagelabels.mat
|    setid.mat
|--- jpg/
```

### [Stanford 120 Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
```
SDog120/
|    file_list.mat
|    test_list.mat
|    train_list.mat
|--- train/
|--- test/
|--- Images/
|--- Annotation/
```

### [MIT 67 Indoor Scenes](http://web.mit.edu/torralba/www/indoor.html)
```
MIT67/
|    TrainImages.txt
|    TestImages.txt
|--- Annotations/
|--- Images/
|--- test/
|--- train/
```

### [Stanford 40 Actions](http://vision.stanford.edu/Datasets/40actions.html)
```
stanford_40/
|    attributes.txt
|--- ImageSplits/
|--- JPEGImages/
|--- MatlabAnnotations/
|--- XMLAnnotations/
```

### [Caltech-UCSD 200 Birds](http://www.vision.caltech.edu/visipedia/CUB-200.html)
Layout should be the following for the dataloader to load correctly

```
CUB_200_2011/
|    README
|    bounding_boxes.txt
|    classes.txt
|    image_class_labels.txt
|    images.txt
|    train_test_split.txt
|--- attributes
|--- images/
|--- parts/
|--- train/
|--- test/
```
