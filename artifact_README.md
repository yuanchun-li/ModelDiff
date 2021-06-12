# ModelDiff: Testing-based DNN Similarity Comparison for Model Reuse Detection

## About
This is the artifact associated with our ISSTA paper "ModelDiff: Testing-based DNN Similarity Comparison for Model Reuse Detection".

ModelDiff is a testing-based approach to deep learning model similarity comparison. Instead of directly comparing the weights, activations, or outputs of two models, ModelDiff compares their behavioral patterns on the same set of test inputs. Specifically, the behavioral pattern of a model is represented as a decision distance vector (DDV), in which each element is the distance between the model's reactions to a pair of inputs. The knowledge similarity between two models is measured with the cosine similarity between their DDVs.
To evaluate ModelDiff, we created a benchmark that contains 144 pairs of models that cover most popular model reuse methods, including transfer learning, model compression, and model stealing. Our method achieved 91.7% correctness on the benchmark, which demonstrates the effectiveness of using ModelDiff for model reuse detection. A study on mobile deep learning apps has shown the feasibility of ModelDiff on real-world models.

## Environment
- Ubuntu 16.04
- CUDA 10.0

## Dependencies
- PyTorch 1.5.0
- TorchVision 0.6.0
- AdverTorch 0.2.0

## Get start
- You should have a GPU on your device because the adversarial sample computation is pretty slow
- You should first install CUDA 10.2 on your device (if you don't have) from [here](https://developer.nvidia.com/cuda-downloads)
- Install [Anaconda](https://www.anaconda.com/) and create a new environment and enter the environment
```
conda create --name modeldiff python=3.6
```
- Install pytorch in the new environment
```
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
```
- Install AdvTorch
```
pip install advertorch
```
- Install other packages

```
pip install scipy
```
- Make a new directory called ``data`` and Download all three datasets listed below in the ``data`` directory 
```
data\
|--- CUB_200_2011/
|--- stanford_dog/
|--- MIT_67/
```



## Prepare dataset

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

### [Stanford 120 Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
```
stanford_dog/
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
MIT_67/
|    TrainImages.txt
|    TestImages.txt
|--- Annotations/
|--- Images/
|--- test/
|--- train/
```

## Prepare models
You can change the size of the benchmark and the number of models to use in benchmark.py. The models used in the paper are MobileNetV2 and ResNet18 trained on Flower102 and StanfordDogs120 datasets. You can add other architectures and datasets the ImageBenchmark class of benchmark.py (line 487 to line 503 as following).
```
# Used in the paper
self.datasets = ['Flower102', 'SDog120']
self.archs = ['mbnetv2', 'resnet18']
# Other archs
# self.datasets = ['MIT67', 'Flower102', 'SDog120']
# self.archs = ['mbnetv2', 'resnet18', 'vgg16_bn', 'vgg11_bn', 'resnet34', 'resnet50']
# For debug
# self.datasets = ['Flower102']
# self.archs = ['resnet18']
```

We also provide the benchmark used in the paper and you can download it from [google drive](https://drive.google.com/file/d/1UfhnPB2V2bpwpWxnne1bodI1cIT3q98c/view?usp=sharing). 

## Evaluation
The code to compare DDV (decision distance vector) model similarity is in evaluate.ipynb. It loads the benchmark models from benchmark.py and compare similarity.

## Authors
- Yuanchun Li (Github ID: ylimit, email: pkulyc@gmail.com)
- Ziqi Zhang (Github ID: ziqi-zhang, email: ziqi_zhang@pku.edu.cn)
- Bingyan Liu (email: lby_cs@pku.edu.cn)
- Ziyue Yang (email: Ziyue.Yang@microsoft.com)
- Yunxing Liu (email: Yunxin.Liu@microsoft.com)

## DOI
We put our code at https://zenodo.org/record/4723301#.YIf-rH0zYUE with a public DOI: 10.5281/zenodo.4723301

## Acknowledgement
Some of the code is referred from [Renofeation](https://github.com/cmu-enyac/Renofeation) and we thank the authors for sharing the code.