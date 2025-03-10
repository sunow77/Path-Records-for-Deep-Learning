# fast.ai

## 0

| 标题 | 日期 | 笔记 |
| :--: | :--: | :--: |
|[官网](https://www.fast.ai/)|-|-|
|[论坛](https://forums.fast.ai/)|-|包含针对各版本课程的模块|
|[Practical Deep Learning for Coders 2022](https://course.fast.ai/)|2022|2022版，2 parts|
|Practical Deep Learning for Coders 2020|2020|2020版，已经被替代|
|[Practical Deep Learning for Coders 2019](https://course19.fast.ai/ )|2019|2019版，2 parts；[hiromis的notes (github.com)](https://github.com/hiromis/notes/tree/master)|
|Practical Deep Learning for Coders 2018|2018|2018版，已经被替代|
|[fast.ai](https://docs.fast.ai/)|-|fastai的documentation|
|[Deep Learning for Coders with fastai and PyTorch: AI Applications Without a PhD]([Deep Learning for Coders with fastai and PyTorch: AI Applications Without a PhD: Howard, Jeremy, Gugger, Sylvain: 9781492045526: Amazon.com: Books](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527))|-|一本书，[免费]([Practical Deep Learning for Coders - The book](https://course.fast.ai/Resources/book.html))|
|[new fast.ai course: A Code-First Introduction to Natural Language Processing – fast.ai](https://www.fast.ai/posts/2019-07-08-fastai-nlp.html)|-|其他类型课程|

## [Practical Deep Learning 2022](https://course.fast.ai/) & [Fastbook](https://github.com/fastai/fastbook)
### 1: Getting started (PDL2022)

#### *Is it a bird? Creating a model from your own data*

*   流程

  1. 下载库，准备好
  2. 生成训练集和验证集
  3. 训练
  4. 验证
* 一些初学

  ```python
  from fastcore.all import * 
  '尽管fastai可以自动处理对fastcore的依赖，使fastai可以使用fastcore的部分功能，但是为了代码更简洁，依然显式地导入了fastcore'
  
  L #转换成fastcore中的唯一一个类L
  dest = (path/o) #path/o
  failed = verify_images(get_image_files(path)) #failed是L类
  ```
  ```python
  from fastai.vision.all import *
  
  Image.open('path').to_thumb(200,200) #Image.open().to_thumb()等比例放大
  download_images(dest, urls=search_images(f'{o} photo')) #download_images
  resize_images(path/o, max_size=200, dest=path/o) #resize_images
  get_image_files(path) #get_image_files
  verify_images(get_image_files(path)) #verify_images
  failed.map(Path.unlink) #failed.map
  dls = DataBlock( #DataBlock
    blocks = (ImageBlock, CategoryBlock), #ImageBlock, CategoryBlock
    get_items = get_image_files, #get_image_files
    splitter = RandomSplitter(valid_pct = 0.2, seed = 42), #RandomSplitter
    get_y = parent_label, #parent_label
    item_tfms = [Resize(200, method = 'squish')] #Resize
  ).dataloaders(path, bs = 32) #dataloaders
  dls.show_batch(max_n=6) #show_batch
  learn = vision_learner(dls, resnet18, metrics = error_rate) #vision_learner
  learn.fine_tune(5) #fine_tune
  cat_or_dog,_,pros_cat = learn.predict(PILImage.create('dog.jpg')) #predict; PILImage.create也可属于fastai，只是它有自己的底层依赖库
  ```

  ```python
  from duckduckgo_search import DDGS '导入库'
  from fastdownload import download_url '用于下载库'
  import time '鉴于效率非常重要，因此要记录time'
  ```

### 1: intro(fastbook)

#### 1.1 初学

```python
#图像识别
path = untar_data(URLs.PETS)/'images' 
#untar_data()从fastai内置的数据库中下载解压数据并进入'images'的文件夹中
#untar_data(URLs.PETS)返回了一个Path对象，是fastai内置库PETS的路径

def is_cat(x): return x[0].isupper() #这个数据集中，Cat是大写，dog是小写，以此来区分猫和狗
dls = ImageDataLoaders.from_name_func( 
    path, 
    get_image_files(path),  #获得所有图像文件
    valid_pct=0.2, seed=42,
    label_func=is_cat,  #返回True或False
    item_tfms=Resize(224) #缩放到224*224像素
)
#ImageDataLoaders.from_name_func是fastai的高级封装，创建数据加载器更加简洁，专门用于从文件名提取标签的任务
#如果使用DataBlock👇
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=is_cat,
    item_tfms=Resize(224)
).dataloaders(path)
#`item_tfms`应用于每个项目（在本例中，每个项目都被调整为 224 像素的正方形），而`batch_tfms`应用于一次处理一批项目的 GPU

learn = vision_learner(dls, resnet34, metrics=error_rate) #error_rate & accuracy
learn.fine_tune(1)
```

```python
from types import SimpleNamespace
uploader = SimpleNamespace(data = ['images/chapter1_cat_example.jpg'])
#相当于uploader = {'data':['images/chapter1_cat_example.jpg']}
```

```python
#Image Classification
#segmentation
path = untar_data(URLs.CAMVID_TINY) #内置库
dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames = get_image_files(path/"images"),
    label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
    codes = np.loadtxt(path/'codes.txt', dtype=str) #这个是储存了分割任务中的类别的文件，比如道路、建筑、汽车等，o就是fnames中的一个元素，o.stem是文件名，o.suffix是扩展名
)
#label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}' #若o是images/cat.jpg，则返回labels/cat_P.jpg

learn = unet_learner(dls, resnet34)
learn.fine_tune(8)

learn.show_results(max_n=6, figsize=(7,8))
```

```python
#NLP
from fastai.text.all import *

dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(4, 1e-2)

learn.predict("I really liked that movie!")
#结果：('pos', tensor(1), tensor([0.0041, 0.9959]))
```

```python
#Tabular
from fastai.tabular.all import *
path = untar_data(URLs.ADULT_SAMPLE)

dls = TabularDataLoaders.from_csv(path/'adult.csv', #这是指定数据文件
                                  path=path, #这是指定数据文件所在的路径
                                  y_names="salary",
    cat_names = ['workclass', 'education', 'marital-status', 'occupation',
                 'relationship', 'race'],
    cont_names = ['age', 'fnlwgt', 'education-num'],
    procs = [Categorify, FillMissing, Normalize]) #这是一个列表，包含对数据进行预处理的步骤。fastai 会自动应用这些预处理步骤。在这个例子中，预处理步骤包括：Categorify: 将分类变量转换为类别类型（通常是整数编码）；FillMissing: 填充缺失值（如果有的话）；Normalize: 对连续变量进行标准化处理（通常是减去均值并除以标准差）。

learn = tabular_learner(dls, metrics=accuracy) #salary是分类是否为高收入者，所以metrics仍用accuracy或error_rate如果是连续变量，则不能使用这个metrics
learn.fit_one_cycle(3) #没有预训练模型，所以不用fine_tune
```

```python
#Collaborative Filtering，协调过滤的结构一般比较简单
#根据用户以前的观影习惯，预测用户可能喜欢的电影
from fastai.collab import *
path = untar_data(URLs.ML_SAMPLE)
dls = CollabDataLoaders.from_csv(path/'ratings.csv') #这是用了默认设置，csv中最后一列是因变量；协同过滤的特征通常也都是离散的分类参数，如果有连续参数，就不能用CollabDataLoaders类简单预测
learn = collab_learner(dls, y_range=(0.5,5.5))
learn.fine_tune(10) #这里使用了fine_tune而不是fit_one_cycle
learn.show_results()
```



#### 1.2 **怎样快速获取fastai中方法的解释-doc**

```python
doc(learn.predict)
'''返回learn.predict方法的解释'''
```

#### 1.3 过拟合

即使您的模型尚未完全记住所有数据，在训练的早期阶段可能已经记住了其中的某些部分。因此，您训练的时间越长，您在训练集上的准确性就会越好；验证集的准确性也会在一段时间内提高，但最终会开始变差，因为模型开始记住训练集而不是在数据中找到可泛化的潜在模式。当这种情况发生时，我们说模型*过拟合*。

我们有很多避免过拟合的办法，但只有真的出现过拟合了才会用这些办法。我们经常看到一些人训练模型，他们有充足的数据，但是过早地使用了避免过拟合的办法，结果导致模型的准确性不好，还不如过拟合了的模型准确性高。

#### 1.4 架构

* CNN：创建计算机视觉模型的当前最先进方法

  ResNet，一种标准架构，有18、34、50、101和152

#### 1.5 预训练/迁移学习

使用预训练模型是我们训练更准确、更快速、使用更少数据和更少时间和金钱的最重要方法。您可能会认为使用预训练模型将是学术深度学习中最研究的领域...但您会非常、非常错误！预训练模型的重要性通常在大多数课程、书籍或软件库功能中**没有**得到认可或讨论，并且在学术论文中很少被考虑。当我们在 2020 年初写这篇文章时，事情刚刚开始改变，但这可能需要一段时间。因此要小心：您与之交谈的大多数人可能会严重低估您可以在深度学习中使用少量资源做些什么，因为他们可能不会深入了解如何使用预训练模型。

使用一个预训练模型来执行一个与其最初训练目的不同的任务被称为***迁移学习***。不幸的是，**由于迁移学习研究不足，很少有领域提供预训练模型**。例如，目前在医学领域很少有预训练模型可用，这使得在该领域使用迁移学习具有挑战性。此外，目前还不清楚如何将迁移学习应用于诸如时间序列分析之类的任务。

#### 1.6 head

When using a pretrained model, `vision_learner` will remove the last layer, since that is always specifically customized to the original training task (i.e. ImageNet dataset classification), and replace it with one or more new layers with randomized weights, of an appropriate size for the dataset you are working with. This last part of the model is known as the *head*.

#### 1.7 训练集、验证集、测试集

在用训练集训练后，我们用验证集查看训练效果，根据验证集的效果，调整超参数hyperparameter，因此，验证集仍然半暴露在训练模型中。为了能更好地评估模型的效果，用验证集显然是不理想的，所以我们还会隔绝出一个完全没有用过的测试集。

当然如果数据量不够，测试集并不一定是必须的，但是当然有是最好的。

在我们实际的训练中，验证集和测试集的选择很有讲究。比如我们预测时间序列，最好的划分是把最近的一段时间作为验证集/测试集，这样我们可以评估模型对未来的预测效果；比如我们识别驾驶员的行为，最好的划分是把一个驾驶员完全隔绝成验证集/测试集，这样我们可以评估模型对不同的人是不是都有很好的识别效果。

#### 1.8 其他

* 时间序列转换成图像

  时间序列数据有各种转换方法。例如，fast.ai 学生 Ignacio Oguiza 使用一种称为**Gramian Angular Difference Field（GADF）**的技术，从一个时间序列数据集中为橄榄油分类创建图像，你可以在图 1-15 中看到结果。然后，他将这些图像输入到一个图像分类模型中，就像你在本章中看到的那样。尽管只有 30 个训练集图像，但他的结果准确率超过 90%，接近最先进水平。

  ![](img/dlcf_0115.png)

  另一个有趣的 fast.ai 学生项目示例来自 Gleb Esman。他在 Splunk 上进行欺诈检测，使用了用户鼠标移动和鼠标点击的数据集。他通过绘制显示鼠标指针位置、速度和加速度的图像，使用彩色线条，并使用[小彩色圆圈](https://oreil.ly/6-I_X)显示点击，将这些转换为图片，如图 1-16 所示。他将这些输入到一个图像识别模型中，就像我们在本章中使用的那样，效果非常好，导致了这种方法在欺诈分析方面的专利！

  ![](img/dlcf_0116.png)

  另一个例子来自 Mahmoud Kalash 等人的论文“使用深度卷积神经网络进行恶意软件分类”，解释了“恶意软件二进制文件被分成 8 位序列，然后转换为等效的十进制值。这个十进制向量被重塑，生成了一个代表恶意软件样本的灰度图像”，如图 1-17 所示。

  ![](img/dlcf_0117.png)

* 术语汇总

  | Term             | Meaning                                                      |
  | ---------------- | ------------------------------------------------------------ |
  | Label            | The data that we're trying to predict, such as "dog" or "cat" |
  | Architecture     | The _template_ of the model that we're trying to fit; the actual mathematical function that we're passing the input data and parameters to |
  | Model            | The combination of the architecture with a particular set of parameters |
  | Parameters       | The values in the model that change what task it can do, and are updated through model training |
  | Fit/拟合         | Update the parameters of the model such that the predictions of the model using the input data match the target labels |
  | Train            | A synonym for _fit_                                          |
  | Pretrained model | A model that has already been trained, generally using a large dataset, and will be fine-tuned |
  | Fine-tune        | Update a pretrained model for a different task               |
  | Epoch            | One complete pass through the input data                     |
  | Loss             | A measure of how good the model is, chosen to drive training via SGD |
  | Metric           | A measurement of how good the model is, using the validation set, chosen for human consumption |
  | Validation set   | A set of data held out from training, used only for measuring how good the model is |
  | Training set     | The data used for fitting the model; does not include any data from the validation set |
  | Overfitting      | Training a model in such a way that it _remembers_ specific features of the input data, rather than generalizing well to data not seen during training |
  | CNN              | Convolutional neural network; a type of neural network that works particularly well for computer vision tasks |

### 2：Deployment(PDL2022)

