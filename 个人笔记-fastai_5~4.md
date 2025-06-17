**<font size=8>fast.ai</font>**

## [Practical Deep Learning 2022](https://course.fast.ai/) & [Fastbook](https://github.com/fastai/fastbook)
### 4: Natural Language (NLP) (PDL2022)

#### 4.1 发展

（1）**ULMFit** (用的RNN): Wikitext(103)Language Model (30%准确率) → IMDb Language Model → IMDb Classifier

（2）**Transformers** (掩码语言建模): 用于机器阅读理解、句子分类、命名实体识别、机器翻译和文本摘要等

（3）...

可以看到似乎Transformer要比ULMFit高级，实际上两者的用途不同；另外，ULMFit能够阅读更长的句子，如果一个document包含超过2000个单词，那么就更推荐使用ULMFit进行分类。

#### 4.2 最重要的package

1/ pandas

2/ numpy

3/ matplotlib

4/ pytorch

##### 参考书：[Python for Data Analysis, 3E About the Open Edition]([Python for Data Analysis, 3E](https://wesmckinney.com/book/))

#### 4.3 Tokenization

A deep learning model expects numbers as inputs, not English sentences! So we need to do two things:

- *Tokenization*: Split each text up into words (or actually, as we'll see, into *tokens*)
- *Numericalization*: Convert each word (or token) into a number.

The details about how this is done actually depend on the particular model we use. So first we'll need to pick a model. There are thousands of models available, but a reasonable starting point for nearly any NLP problem is to use this：

`microsoft/deberta-v3-small`

```python
tokz.tokenize("A platypus is an ornithorhynchus anatinus.")
'''
['▁A',
 '▁platypus',
 '▁is',
 '▁an',
 '▁or',
 'ni',
 'tho',
 'rhynch',
 'us',
 '▁an',
 'at',
 'inus',
 '.']
'''
```

#### 4.4 训练集与验证集的划分

Training set & Validation set

In practice, a random split like we've used here might not be a good idea -- here's what Dr Rachel Thomas has to say about it:

> "*One of the most likely culprits for this disconnect between results in development vs results in production is a poorly chosen validation set (or even worse, no validation set at all). Depending on the nature of your data, choosing a validation set can be the most important step. Although sklearn offers a `train_test_split` method, this method takes a random subset of the data, which is a poor choice for many real-world problems.*"

I strongly recommend reading her article [How (and why) to create a good validation set](https://www.fast.ai/2017/11/13/validation-sets/) to more fully understand this critical topic.

* 随机划分数据集不适用的情况：时间序列、新人脸、新的船等等；
* cross-validation比较危险，除非用到的case是那种可以随机洗牌的情况（随机分ABC三组数据集，AB合并做训练集-C做验证集，三组循环，最后求平均值作为模型的performance）；
* 所以用一个test set测试集去最终确认一下模型的好坏也蛮重要的。

#### 4.5 Metrics

In real life, outside of Kaggle, things not easy... As my partner Dr Rachel Thomas notes in [The problem with metrics is a big problem for AI](https://www.fast.ai/2019/09/24/metrics/):

> At their heart, what most current AI approaches do is to optimize metrics. The practice of optimizing metrics is not new nor unique to AI, yet AI can be particularly efficient (even too efficient!) at doing so. This is important to understand, because any risks of optimizing metrics are heightened by AI. While metrics can be useful in their proper place, there are harms when they are unthinkingly applied. Some of the scariest instances of algorithms run amok all result from over-emphasizing metrics. We have to understand this dynamic in order to understand the urgent risks we are facing due to misuse of AI.

* Metrics很多时候并不是*我们真正关心的事情*，它只是*我们关心的事情*的一个代理，如我们关心老师的教学效果，metrics是学生的分数；
* Metrics会被故意地、作弊地拉高，使它失去了衡量*我们关心的事情*的能力，如老师人为拉高学生的分数，它不再能反映老师的教学效果；
* Metrics会更短视，比如银行一旦将cross-selling这个metrics作为目标，就会催生出各种虚假开户，而实际上银行的目标是维护良好的客户关系，后者才是长远的战略，比如*我们关心的事情*是提高视频影响力，用了点击率作为Metrics，就没有考虑到一些视频长期来看对读者的帮助和塑造；
* 很多Metrics是在一个高度成瘾的环境收集数据的，比如数据收集到小朋友喜欢吃甜食，算法会让食物越来越甜，永远不可能output出有营养的食物；
* 尽管如此Metrics依然很有用，需要考虑多个metrics来避免上述问题，但最终我们要努力将它们整合；
* Metrics通过定量方式衡量结果，但我们依然需要定性的信息才能获得好的metrics；
* 去询问已在此山中的人永远可以foresee一些不良后果，如老师可以很容易地知道，用学生分数作为唯一衡量标准会导致什么糟糕的结果。

#### 4.6 codes

```python
# 检查是否为kaggle环境
import os
iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
iskaggle

# 下载datasets
from pathlib import Path
if iskaggle:
    path = Path('../input/us-patent-phrase-to-phrase-matching')
    ! pip install -q datasets #只需要跑一次就到了页面中了
    
# ！表示后面的不是python命令，是shell命令，但是如果想利用到python中的参数，就用{}框起来
!ls {path}
```

```python
#数据的预处理工作
# 定义一个训练集的dataframe
import pandas as pd
df = pd.read_csv(path/'train.csv')
df.describe(include='object') #一个非常重要的dataframe方法

# 构建一个df
df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC1: ' + df.anchor
df.input.head()
'''
0    TEXT1: A47; TEXT2: abatement of pollution; ANC...
1    TEXT1: A47; TEXT2: act of abating; ANC1: abate...
2    TEXT1: A47; TEXT2: active catalyst; ANC1: abat...
3    TEXT1: A47; TEXT2: eliminating process; ANC1: ...
4    TEXT1: A47; TEXT2: forest region; ANC1: abatement
Name: input, dtype: object
'''
df['input'][0], df.input[0]
''TEXT1: A47; TEXT2: abatement of pollution; ANC1: abatement''
            
# 实例化一个训练集的Dataset
from datasets import Dataset,DatasetDict
ds = Dataset.from_pandas(df)
'''
Dataset({
    features: ['id', 'anchor', 'target', 'context', 'score', 'input'],
    num_rows: 36473
})
其中input是整个string句子
'''

# 选择一个model,就有了和这个model对应的vocabulary，然后实例化它的tokz工具
model_nm = 'microsoft/deberta-v3-small'
from transformers import AutoModelForSequenceClassification,AutoTokenizer
tokz = AutoTokenizer.from_pretrained(model_nm)

# 定义一个用来tokz某段文字的函数，并应用
def tok_func(x): return tokz(x["input"])
toknum_ds = ds.map(tok_func, batched=True) #来自HuggingFace的datasets库,这使得它不再只有'input'还有'input_ids'
'''
.map(tok_func, batched=True)：对 ds 中的数据应用 tok_func 进行映射（map），并且启用批处理（batched=True），以提高效率。
tok_ds：返回一个新的 Dataset，其中每个文本已经被 tok_func 处理过（通常是 Tokenizer）
'''
#row = toknum_ds[0]
#row['input'], row['input_ids'] '①第一句话string，②一串数字'
#tokz.vocab['▁of'] #这里有一个vocabulary，tocken→数字
#'265'

# 准备一个lables，transformer一向都认为labels就说有一列叫做labels，所以得改名字
toknum_ds = toknum_ds.rename_columns({'score':'labels'})

# 把之前已经数字化了的Dataset分一下，分成一个训练集一个验证集
dds = toknum_ds.train_test_split(0.25, seed=42)

# 定义Metrics
import numpy as np
def corr(x,y): return np.corrcoef(x,y)[0][1]
def corr_d(eval_pred): return {'pearson': corr(*eval_pred)} #eval_pred通常是包含预测值和实际值的元组或列表，*表示将预测值和实际值拆解成位置参数传递给corr() [另外，这个'pearson'标签最后会出现在训练结果里]
```

```python
#训练
from transformers import TrainingArguments,Trainer
bs = 128
epochs = 4
lr = 8e-5
args = TrainingArguments(
    'outputs',                # 输出目录：训练过程中模型和日志将保存在此目录中
    learning_rate=lr,         # 学习率：模型优化的步长（lr 是事先定义好的变量）
    warmup_ratio=0.1,         # 预热比例：学习率从 0 线性增加到初始学习率的比例（0.1 表示 10% 的训练步数作为预热阶段）
    lr_scheduler_type='cosine',  # 学习率调度器类型：使用余弦退火（Cosine Annealing）策略来逐步降低学习率
    fp16=True,                # 是否使用半精度训练：将训练过程中的浮点数精度降为 16 位，可以提高训练速度并减少显存占用
    evaluation_strategy="epoch",  # 评估策略：每个训练轮（epoch）结束后进行评估
    per_device_train_batch_size=bs,  # 每个设备（GPU）上训练的批量大小（`bs` 是事先定义的变量）
    per_device_eval_batch_size=bs*2,  # 每个设备上评估的批量大小，通常评估时批量可以稍大一些
    num_train_epochs=epochs,  # 训练轮数：训练过程中会进行的完整数据集迭代次数（`epochs` 是事先定义的变量）
    weight_decay=0.01,        # 权重衰减：L2 正则化的强度，用于防止过拟合
    report_to='none' # 禁用报告（如不报告到 TensorBoard 或 WandB），如果需要报告，可以设置为 'tensorboard' 或 'wandb'
)
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1) #将用的模型
trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'], tokenizer=tokz,
                  compute_metrics=corr_d) #将model、超参数们、data整合在一起的类

trainer.train()
```

```python
#测试集
# 定义一个测试集的dataframe，构建一个eval_df，基于这实例化一个eval_ds并将它tokenization
eval_df = pd.read_csv(path/'test.csv')
eval_df.describe()
eval_df['input'] = 'TEXT1: ' + eval_df.context + '; TEXT2: ' + eval_df.target + '; ANC1: ' + eval_df.anchor
eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)

# 预测测试集
preds = trainer.predict(eval_ds).predictions.astype(float)
preds = np.clip(preds, 0, 1) #将<0>1的数值规整到0~1

#将结果导出到csv
import datasets
submission = datasets.Dataset.from_dict({
    'id': eval_ds['id'],
    'score': preds
})
submission.to_csv('submission.csv', index=False)
```

#### 4.7 超参数：权重衰减weight_decay

L2 正则化通过在损失函数中加入一个与模型权重的平方和成正比的项来实现惩罚。具体来说，假设我们有一个损失函数 `L(w)`，表示模型的损失，其中 `w` 是模型的权重参数，那么加入 L2 正则化后的损失函数 `L2(w)` 就是：
$$
L2(w)=L(w) + \lambda \sum_{i} w_i^2
$$
其中：

- `L(w)` 是原始的损失函数（如交叉熵、均方误差等）。
- `w_i` 是模型的第 `i` 个权重。
- `λ` 是正则化强度的超参数，控制 L2 正则化的影响。较大的 `λ` 会对模型的训练产生较大的影响。

**作用**

- **限制权重的大小**：L2 正则化鼓励模型的权重 `w` 尽可能小，避免出现过大的权重值，这样可以减少模型的复杂度，防止过拟合。
- **平滑模型**：通过抑制大权重，L2 正则化促使模型学习到更加平滑的函数，而非过于复杂的、过拟合训练数据的函数。
- **改进泛化能力**：通过防止过拟合，L2 正则化使得模型在未见过的数据（测试集）上的表现更加稳定。

### 4: Chapter 10: nlp (fastbook10)

#### 4.1 自监督学习

使用嵌入在自变量中的标签来训练模型，而不是需要外部标签。例如，训练一个模型来预测文本中的下一个单词。自监督学习也可以用于其他领域；例如，参见[“自监督学习和计算机视觉”](https://oreil.ly/ECjfJ)以了解视觉应用。

自监督学习通常不用于直接训练的模型，而是用于预训练用于迁移学习的模型。

* 通用语言模型微调（ULMFiT）方法：有一个基于维基百科语料库的预训练模型，用IMDb的语料库进行微调，再进行情感分类

![](D:\Git\a\Path-Records\img\dlcf_1001.png)

* 分词方法：基于单词的、基于子词的和基于字符的

#### 4.2 Tokenization & Numericalization

由分词过程创建的列表的一个元素。它可以是一个单词*word tokenization*，一个单词的一部分（一个*子词*）*subword tokenization*，或一个单个字符。

##### 4.2.1 Word Tokenization

```python
from fastai.text.all import *
path = untar_data(URLs.IMDB)
path.ls()
# 获取path中指定folders中的files
files = get_text_files(path, folders=['train', 'test', 'unsup'])
txt = files[0].open().read()
txt[:100]
```

```python
# WordTokenizer是一个分词器类，实例化WordTokenizer类
spacy = WordTokenizer()
# first()作用是返回列表的第一个元素。这里它取的是spacy([txt])处理后返回的第一个Doc对象的token列表。分词器接受文档集合，所以用[txt]
toks = first(spacy([txt]))
# 显示`collection`的前`n`个项目，以及完整的大小——这是`L`默认使用的
print(coll_repr(toks,30))
```

```python
# 通过Tokenizer类增加额外的功能
tkn = Tokenizer(spacy)
print(coll_repr(tkn(txt),30))
'''
(#207) ['xxbos','xxmaj','once','again','xxmaj','mr','.','xxmaj','costner','has','dragged','out','a','movie','for','far','longer','than','necessary','.','xxmaj','aside','from','the','terrific','sea','rescue','sequences',',','of','which'...]
'''
```

👆一些特殊标记：

| 主要特殊标记 | 代表                                                         |
| ------------ | ------------------------------------------------------------ |
| xxbos        | 指示文本的开始，意思是“流的开始”）。通过识别这个开始标记，模型将能够学习需要“忘记”先前说过的内容，专注于即将出现的单词。 |
| xxmaj        | 指示下一个单词以大写字母开头（因为减小vocabulary的体量，节省计算和内存资源，我们将所有字母转换为小写） |
| xxunk        | 指示下一个单词是未知的                                       |

##### 4.2.2 Subword Tokenization

```python
# 读取200个files中的句子txts
txts = L(o.open().read() for o in files[:200])
# 将句子txts拆分成sz个vocabulary，并tokenize txt
def subword(sz):
    sp = SubwordTokenizer(vocab_sz=sz)
    sp.setup(txts)
    return ' '.join(first(sp([txt]))[:40])
# 分成1000个vocabulary
subword(1000)
'▁ J ian g ▁ X ian ▁us es ▁the ▁comp le x ▁back st or y ▁of ▁L ing ▁L ing ▁and ▁Ma o ▁D a o b ing ▁to ▁st ud y ▁Ma o \' s ▁" c'
# 分成100个vocabulary，为了能拆分，好多都拆成字母了
subword(100)
'▁ J i a n g ▁ X i a n ▁ u s e s ▁the ▁ c o m p l e x ▁ b a c k s t o r y ▁ o f ▁ L'
```

##### 4.2.3 Numericalization

```python
# 将txts前200条text都Word Tokenization（也可以subword tokenization，本例用了前者）
toks200 = txts[:200].map(tkn)
toks200[0]
'''
获得了一个分词后的列表，列表长度为200
(#158) ['xxbos','xxmaj','jiang','xxmaj','xian','uses','the','complex','backstory','of','xxmaj','ling','xxmaj','ling','and','xxmaj','mao','xxmaj','daobing','to'...]
'''

# 类比上面的subword(),因为要手动建立个vocabulary，这个实例化后也要setup一下
num = Numericalize()
num.setup(toks200) #基于分词后的结果设置数字映射
nums = num(toks)[:20]
'''
TensorText([   0,    0, 1269,    9, 1270,    0,   14,    0,    0,   12,    0,
               0,   15, 1271,    0,   22,   24,    0,  795,   24])
'''
# 将数字化的句子再映射回tokens
' '.join(num.vocab[o] for o in nums)
'''
'xxunk xxunk uses the complex xxunk of xxunk xxunk and xxunk xxunk to study xxunk \'s " xxunk revolution "'
'''
```

`Numericalize`的默认值为`min_freq=3`和`max_vocab=60000`。`max_vocab=60000`导致 fastai 用特殊的*未知单词*标记`xxunk`替换除最常见的 60,000 个单词之外的所有单词。这有助于避免过大的嵌入矩阵，因为这可能会减慢训练速度并占用太多内存，并且还可能意味着没有足够的数据来训练稀有单词的有用表示。然而，通过设置`min_freq`来处理最后一个问题更好；默认值`min_freq=3`意味着出现少于三次的任何单词都将被替换为`xxunk`。

##### 4.2.4 将这些txt放进batches里面，形成DataLoader

<img src="D:\Git\a\Path-Records\img\04-2-4.jpg" style="zoom:100%;" />

步骤：

- 将200条txt分词后映射成数字，再将200条数字拼接成一个stream
- 两种情况
  - stream切成固定长度的mini-stream，并讲它们分成固定大小的batch，这样batch1中的mini-stream1和mini-stream2连续
  - stream reshape成规整的二维结构，然后切成不同的batch（如上图），这样batch1中的mini-stream1和batch2中mini-stream1连续

```python
# toks200是200条txt分词后的，num200就是200条分词句子映射成数字的
num200 = toks200.map(num)
# DataLoader
dl = LMDataLoader(num200)
x,y = first(dl)
x.shape, y.shape
'(torch.Size([64, 72]), torch.Size([64, 72]))，可见DataLoader将stream拆成了64个mini-stream，每个mini-stream有72个tokens'
# x和y只是相差一个token
' '.join(num.vocab[o] for o in x[0][:15])
'xxbos xxmaj xxunk xxmaj xxunk uses the complex xxunk of xxmaj xxunk xxmaj xxunk and'
' '.join(num.vocab[o] for o in y[0][:15])
'xxmaj xxunk xxmaj xxunk uses the complex xxunk of xxmaj xxunk xxmaj xxunk and xxmaj'
```

#### 4.3 训练文本分类器

* 使用迁移学习训练最先进的文本分类器有两个步骤：首先，我们需要微调在 Wikipedia 上预训练的语言模型以适应 IMDb 评论的语料库，然后我们可以使用该模型来训练分类器。

##### 4.3.1 语言识别-数据加载器DataBlock

* **实例方法**，需要实例化类，然后才能调用的方法，MyClass.instance_method()会报错；**类方法**就不需要实例化类，直接调用MyClass.class_method()不会报错，而且可以访问类变量；**静态方法**也不需要实例化类，直接调用MyClass.static_method()也不会报错，但没办法访问类变量。

```python
from fastai.text.all import *
path = untar_data(URLs.IMDB)
# 这是上面全部手动代码的汇总--------------------------------------------------------------------------------
files = get_text_files(path, folders=['train', 'test', 'unsup'])
#txt = files[0].open().read()
spacy = WordTokenizer()
#toks = first(spacy([txt]))
tkn = Tokenizer(spacy)
# 读取200个files中的句子txts
txts = L(o.open().read() for o in files[:200])
# 将句子txts拆分成sz个vocabulary，并tokenize txt
#def subword(sz):
#    sp = SubwordTokenizer(vocab_sz=sz)
#    sp.setup(txts)
#    return ' '.join(first(sp([txt]))[:40])
#tks = tkn(txt)
toks = txts.map(tkn)
num = Numericalize()
num.setup(toks)
#nums = num(toks)[:20]
num = toks.map(num)
dl = LMDataLoader(num) #没法指定batch_size
x,y = first(dl)
# fastai有现成的方法---------------------------------------------------------------------------------------
get_imdb = partial(get_text_files, folders=['train','test','unsup'])
# 语言模型的数据加载器
dls_lm = DataBlock(
    blocks = TextBlock.from_folder(path,is_lm=True),
    get_items = get_imdb,
    splitter = RandomSplitter(0.1)
).dataloaders(path, path=path, bs=128, seq_len=80)
dls_lm.show_batch(max_n=3)
```

|      | text                                                         | text_                                                        |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 0    | xxbos xxmaj it ’s awesome ! xxmaj in xxmaj story xxmaj mode , your going from punk to pro . xxmaj you have to complete goals that involve skating , driving , and walking . xxmaj you create your own skater and give it a name , and you can make it look stupid or realistic . xxmaj you are with your friend xxmaj eric throughout the game until he betrays you and gets you kicked off of the skateboard | xxmaj it ’s awesome ! xxmaj in xxmaj story xxmaj mode , your going from punk to pro . xxmaj you have to complete goals that involve skating , driving , and walking . xxmaj you create your own skater and give it a name , and you can make it look stupid or realistic . xxmaj you are with your friend xxmaj eric throughout the game until he betrays you and gets you kicked off of the skateboard xxunk |
| 1    | what xxmaj i ‘ve read , xxmaj death xxmaj bed is based on an actual dream , xxmaj george xxmaj barry , the director , successfully transferred dream to film , only a genius could accomplish such a task . \n\n xxmaj old mansions make for good quality horror , as do portraits , not sure what to make of the killer bed with its killer yellow liquid , quite a bizarre dream , indeed . xxmaj also , this | xxmaj i ‘ve read , xxmaj death xxmaj bed is based on an actual dream , xxmaj george xxmaj barry , the director , successfully transferred dream to film , only a genius could accomplish such a task . \n\n xxmaj old mansions make for good quality horror , as do portraits , not sure what to make of the killer bed with its killer yellow liquid , quite a bizarre dream , indeed . xxmaj also , this is |

##### 4.3.2 语言识别-Fine-tune

* **Embedding**：嵌入是把文字转换成计算机能理解的数字，而且这种转换不是简单的 1 对 1 映射，而是让语义相近的词在数值空间里也靠得更近。常见的 NLP 任务都会用到 Embedding，比如：**Word2Vec**（Google 开发的词向量模型）、**GloVe**（斯坦福开发的词向量）、**FastText**（Facebook 开发的词向量）、**BERT / GPT**（现代 NLP 模型的底层都会用更高级的 Embedding）。

```python
learn = language_model_learner(
    dls_lm,
    AWD_LSTM,
    drop_mult=0.3,      # Dropout 乘数（控制正则化的程度）
    metrics=[accuracy, Perplexity()]     # 评估指标：准确率 + 困惑度（Perplexity）
).to_fp16()      # 将模型转换为半精度（FP16），提升训练速度
```

* **损失函数**：交叉熵损失
* **Perplexity** metrics常用在NLP中，它是损失函数的指数（即torch.exp(cross_entropy)）

* **Dropout**是一种防止神经网络过拟合的方法。它的基本思想是：在训练过程中，随机“丢弃”（设为 0）一部分神经元的输出，防止模型过度依赖某些特定的特征。

###### ①**Dropout vs. Weight Decay：区别对比**

| 特性        | Dropout                                              | Weight Decay (L2 正则化)             |
| ---------------- | -------------------------------------------------------- | ---------------------------------------- |
| 作用方式       | 随机丢弃部分神经元，让网络学会不同的特征             | 限制权重大小，防止过拟合             |
| 适用范围    | 更适用于深度神经网络（尤其是 CNN、RNN、Transformer） | 适用于几乎所有机器学习模型           |
| 训练与测试   | 只在训练时生效，测试时关闭                           | 训练和测试时都生效                   |
| 数学公式     | 让部分神经元的输出设为 0                                 | 给损失函数增加 λ∑w^2 惩罚项              |
| 直观理解     | 让神经网络变成一个小型集成学习                       | 减少大权重，防止模型过度拟合特定数据 |
| 对计算的影响 | 增加计算量，因为每次训练都要随机丢弃不同神经元       | 不会增加计算量                       |

###### ②**什么时候用 Dropout？什么时候用 Weight Decay？**

虽然它们的实现方式不同，但目的都是 防止模型对训练数据过拟合，提高泛化能力。

✔ 可以一起使用！

- Dropout 主要作用在 网络结构 层面（丢弃神经元）
- Weight Decay 主要作用在 参数优化 层面（约束权重大小）
- 现代深度学习模型通常 两者都用

| 情况                                   | 更适合 Dropout    | 更适合 Weight Decay |
| ------------------------------------------ | --------------------- | ----------------------- |
| 数据量小，容易过拟合                   | ✅                     | ✅                       |
| 神经网络很深（CNN、LSTM、Transformer） | ✅                     | ✅                       |
| 参数量少（小型模型，如线性回归）       | ❌                     | ✅                       |
| 过拟合严重                             | ✅（提高 Dropout 率）  | ✅（增大 Weight Decay）  |
| 梯度消失问题（RNN）                    | ✅（Dropout 也能帮忙） | ❌                       |

💡 经验法则：

- 大模型（CNN/RNN） → 两者都用，dropout=0.3~0.5 + wd=1e-4
- 小模型（线性回归） → 主要用 Weight Decay
- 数据量特别小 → Dropout 可以少用，但 Weight Decay 仍然有效

**训练**：

像`vision_learner`一样，当使用预训练模型（这是默认设置）时，`language_model_learner`在使用时会自动调用`freeze`。因此这将仅训练嵌入层，其他部分的权重是被冻结的。之所以只训练嵌入层，是因为在 IMDb 语料库中，可能会有一些词汇在预训练模型的词表中找不到，这些词的嵌入（embeddings）需要随机初始化，并在训练过程中进行优化，而预训练模型的其他部分已经有了较好的参数，因此暂时不会被调整。

fine_tune不会保存半成品模型结果，所以我们用了fit_one_cycle

```python
learn.fit_one_cycle(1, 2e-2)
```

###### ③**fit vs. fit_one_cycle 对比**

| 对比项     | fit                      | fit_one_cycle               |
| ---------- | ------------------------ | --------------------------- |
| 学习率调度 | 固定学习率               | 动态调整（warm-up + decay） |
| 动量调度   | 不变                     | 自适应调整                  |
| 适用场景   | 小规模训练、简单任务     | 深度学习、大规模训练        |
| 优点       | 简单、稳定               | 提高泛化能力、收敛更快      |
| 缺点       | 可能训练慢，泛化能力不佳 | 需要调整参数，稍复杂        |

##### 4.3.3 语言识别-保存模型

```python
# 保存经历1次epoch的模型状态
learn.save('1epoch')
# 加载模型
learn = learn.load('1epoch')
# 初始训练完成，解冻后继续微调模型
learn.unfreeze()
learn.fit_one_cycle(10,2e-3)
# 保存编码器encoder：编码器就是我们训练的所有模型，除了最后一层（它将activations转换成预测token的概率）
learn.save_encoder('finetuned')
```

此时已经完成第二阶段：

![](D:\Git\a\Path-Records\img\dlcf_1001.png)

```python
# Kaggle中将训练好的模型下载下来
path1 = Path('/kaggle/working/')
learn.save(path1/'mymodel')  # 保存模型
learn.load(path1/'mymodel') #加载模型

path2 = Path('/kaggle/working/')
learn.save_encoder(path2/'finetuned')
learn.load_encoder(path2/'finetuned')
```

###### fastai的快捷构造器learn

不同任务有不同的快捷构造器：

| 任务                 | 快捷方法                                            | 底层类    |
| -------------------- | --------------------------------------------------- | --------- |
| 计算机视觉           | `vision_learner`                                    | `Learner` |
| 自然语言处理         | `language_model_learner`、`text_classifier_learner` | `Learner` |
| 表格数据             | `tabular_learner`                                   | `Learner` |
| 协同过滤（推荐系统） | `collab_learner`                                    | `Learner` |

##### 4.3.4 文本生成（不在阶段中）

```python
TEXT = "I liked this movie because"
N_WORDS = 40
N_SENTENCES = 2
# temperature=0.75：控制随机性。>1更随机更有创造力但可能没意义；<1更确定更合理但可能较无聊
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75)
         for _ in range(N_SENTENCES)]
print("\n".join(preds))
```

##### 4.3.5 文本分类-数据加载器DataBlock

```python
# 创建数据加载器
dls_clas = DataBlock(
    blocks=(TextBlock.from_folder(path,vocab=dls_lm.vocab),CategoryBlock),
    '''使用 dls_lm.vocab 的词汇表对文本进行数值化，以保持和 dls_lm 相同的词索引编号，这个词汇表用于将文本转换为模型可处理的数字序列，1 token→1 integer
    通过传递 `is_lm=False`（或者根本不传递 `is_lm`，因为它默认为 `False`），我们告诉 `TextBlock` 我们有常规标记的数据，而不是将下一个标记作为标签'''
    get_y=parent_label,
    get_items=partial(get_text_files,folders=['train','test']),
    splitter=GrandparentSplitter(valid_name='test')
).dataloaders(path,path=path,bs=128,seq_len=72)
dls_clas.show_batch(max_n=4)
```

```python
# 只为了说明怎么处理在分类算法中text长短不一的问题
files = get_text_files(path, folders=['train','test','unsup'])
txts = L(o.open().read() for o in files[:200])
spacy = WordTokenizer()
tkn = Tokenizer(spacy)
toks200 = txts.map(tkn)
num = Numericalize()
num.setup(toks200)
nums_samp = toks200[:10].map(num)
nums_samp.map(len)
'(#10) [158,319,181,193,114,145,260,146,252,295]：可见L类每条大小不等'
```

但是，PyTorch 的 DataLoader需要将批次中的所有项目整合到一个张量中，而一个张量具有固定的形状（即，每个轴上都有特定的长度，并且所有项目必须一致），为了让它们相等，就得填充。

* 填充：我们将扩展最短的文本以使它们都具有相同的大小。为此，我们使用一个特殊的填充标记，该标记将被我们的模型忽略。此外，为了避免内存问题并提高性能，我们将大致相同长度的文本批量处理在一起。我们在每个epoch前（对于训练集）按长度对文档进行排序，分成几个batches，使用每个batch中最大文档的大小作为这个batch的目标大小。

* 使用DataBlock+is_lm=False时，它会自动帮我们操作。

###### DataBlock & DataLoaders

| **组件**         | **作用**                            |
| ---------------- | ----------------------------------- |
| `DataBlock`      | 只是 数据处理的模板，不包含数据     |
| `.dataloaders()` | 将 `DataBlock` 转为 `DataLoaders`   |
| `DataLoaders`    | 真正的数据加载器，包含训练/验证数据 |

###### DataLoaders的主要变种

| 变种名称                  | 适用任务                                |
| ------------------------- | --------------------------------------- |
| `ImageDataLoaders`        | 图像分类（自动从文件夹加载图像）        |
| `TextDataLoaders`         | 自然语言处理（NLP）（自动处理文本数据） |
| `TabularDataLoaders`      | 表格数据（用于结构化数据，如 CSV）      |
| `SegmentationDataLoaders` | 图像分割（像素级分类任务）              |
| `DataLoaders`（基类）     | 通用数据加载器，适用于自定义数据        |

##### 4.3.6 文本分类-Fine-tune

```python
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5,metrics=accuracy).to_fp16()
learn = learn.load_encoder('/kaggle/input/finetuned/pytorch/default/1/finetuned')
```

* 逐层解冻gradual unfreezing：冻结模型的部分层，先训练模型的最后一层，然后逐渐解冻前面的层。从最后一层开始训练，因为这些层已经包含了大部分的特征信息。随着训练的进行，逐渐解冻前面的层，使得模型能够学习更复杂的特征。
* 逐层解冻的原因：①预训练的特征提取层-在大规模数据集上预训练的深度神经网络通常已经学到了通用的特征。我们不需要从头开始训练这些层，而是冻结它们，专注于训练新任务的最后几层。②避免过拟合-如果一次性解冻所有层，模型可能会快速过拟合，因为较小的数据集无法支持所有层的训练。逐层解冻有助于逐渐适应新任务，减少过拟合的风险。③训练效率-冻结前面几层后，计算量减少，训练速度加快。

```python
learn.fit_one_cycle(1,2e-2) #默认模型会冻结，因此只是训练最后一层
# 训练倒数2层
learn.freeze_to(-2)
learn.fit_one_cycle(1,slice(1e-2/(2.6**4),1e-2))
# 训练倒数3层
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))
# 训练所有层
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3))
```

## 5: From-scratch model (Tabular) (PDL2022)

### 5.1 清洗数据

- 一些起手code，你应该已经熟悉

```python
import os
from pathlib import Path

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
if iskaggle: path = Path('../input/titanic')
else:
    path = Path('titanic')
    if not path.exists():
        import zipfile,kaggle
        kaggle.api.competition_download_cli(str(path))
        zipfile.ZipFile(f'{path}.zip').extractall(path)
        
import torch, numpy as np, pandas as pd
np.set_printoptions(linewidth=140)
torch.set_printoptions(linewidth=140, sci_mode=False, edgeitems=7)
pd.set_option('display.width', 140)
```

- 数据清洗

```python
df = pd.read_csv(path/'train.csv')
#检查missing data
df.isna().sum()
 #mode计算众数
modes = df.mode().iloc[0]
df.fillna(modes, inplace=True)
df.isna().sum()
#以上，处理了所有的missing data
```

#### imputing missing values

填补缺失值，在Jeremy的例子中，他使用了众数去填补缺失值，他说这个方法通常有用。一般当数据中包含缺失值，不建议直接删除列，因为并没有实际的理由删除列，但有很多理由保留列。有另一种处理方式，就是新增一列去说明某列是否是缺失值，这样信息就得以保全。

用众数填补缺失值是最最简单的方法，为什么要用这么简单的方法呢？因为大部分情况下，用什么方法填补缺失值都没有什么太大的区别，所以在刚开始建立一个baseline的时候，没有必要搞得太复杂。

```python
#大致review一下包含数字的数据
df.describe(include=(np.number))
df['Fare'].hist();
#发现Fare是个长尾数据，一般我们不太喜欢长尾数据，会影响模型效果，不过有个几乎肯定成果的办法，可以讲长尾数据转换成离散数据
df['LogFare'] = np.log(df['Fare']+1)
#发现Pclass看上去不像个正经数据，像个分类数据，确认一下
pclasses = sorted(df.Pclass.unique())

#大致review一下不是数字的数据：Name、Sex、Ticket、Cabin、Embarked，可以看到有几个数据可以分类为几类，有机会转换成可用数据
df.describe(include=[object])
#想用的分类数据显然不能就用object训练，我们将它转换成dummy column
df = pd.get_dummies(df, columns=["Sex","Pclass","Embarked"])
df.columns
'''Index(['PassengerId', 'Survived', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'LogFare', 'Sex_female', 'Sex_male',
       'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S'],
      dtype='object')'''
added_cols = ['Sex_male', 'Sex_female', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
df[added_cols].head()

#现在可以构建数据了
from torch import tensor
t_dep = tensor(df.Survived) #output
indep_cols = ['Age', 'SibSp', 'Parch', 'LogFare'] + added_cols #input
t_indep = tensor(df[indep_cols].values, dtype=torch.float) #pytorch需要数据是float
t_indep.shape
len(t_indep.shape) #rank
```

### 5.2 构建线性模型⛔

- 初始化parameters

```python
torch.manual_seed(442) #这个会使随机生成数可再现，不过Jeremy在实际操作中并不会使用它
n_coeff = t_indep.shape[1]
coeffs = torch.rand(n_coeff)-0.5 #torch.rand()生成0~1的数，所以减0.5
```

- 利用broadcasting进行计算并求loss

```python
#归一化
vals,indices = t_indep.max(dim=0)
t_indep = t_indep / vals
#计算preds
preds = (t_indep*coeffs).sum(axis=1)
#计算loss
loss = torch.abs(preds-t_dep).mean()
#将上述计算写成函数
def calc_preds(coeffs, indeps): return (indeps*coeffs).sum(axis=1)
def calc_loss(coeffs, indeps, deps): return torch.abs(calc_preds(coeffs, indeps)-deps).mean()
```

### 5.3 梯度下降(one step)⛔

```python
#开启梯度跟踪
coeffs.requires_grad_()
#计算loss
loss = calc_loss(coeffs, t_indep, t_dep)
#反向传播
loss.backward()
coeffs.grad #查看
#手动
with torch.no_grad(): #暂时关闭PyTorch的自动求导机制
    coeffs.sub_(coeffs.grad * 0.1)
    coeffs.grad.zero_()
    print(calc_loss(coeffs, t_indep, t_dep))
```

### 5.4 训练模型

- 清洗好数据，并归一化
- 拆分数据为训练集和验证集

```python
from fastai.data.transforms import RandomSplitter
trn_split,val_split=RandomSplitter(seed=42)(df) #trn_split/val_split是df的rank-0大小的index
'(#10) [788,525,821,253,374,98,215,313,281,305]一共有713rows，随机选取row的数'
trn_indep,val_indep = t_indep[trn_split],t_indep[val_split]
trn_dep,val_dep = t_dep[trn_split],t_dep[val_split]
len(trn_indep),len(val_indep)
```

- 写更新梯度的函数（思路可以参考5.3 with下面的部分）

```python
def update_coeffs(coeffs, lr):
    coeffs.sub_(coeffs.grad * lr)
    coeffs.grad.zero_()
```

- 写训练一个epoch的函数（思路可以参考5.2loss/5.3）

```python
#5.2构建的模型挪到这里
def calc_preds(coeffs, indeps): return (indeps*coeffs).sum(axis=1)
def calc_loss(coeffs, indeps, deps): return torch.abs(calc_preds(coeffs, indeps)-deps).mean()
#5.3写一个epoch
def one_epoch(coeffs, lr):
    loss = calc_loss(coeffs, trn_indep, trn_dep) #引用函数
    loss.backward()
    with torch.no_grad(): update_coeffs(coeffs, lr) #引用函数
    print(f"{loss:.3f}", end="; ")
```

- 初始化参数（参考5.2/5.3）

```python
def init_coeffs(): return (torch.rand(n_coeff)-0.5).requires_grad_()
```

- 写训练n个epoch的函数，输出loss变化，返回参数

```python
def train_model(epochs=30, lr=0.01):
    torch.manual_seed(442) #为了可复现
    coeffs = init_coeffs()
    for i in range(epocks): one_epoch(coeffs, lr=lr)
    return coeffs
```

- 训练

```python
coeffs = train_model(18, lr=0.02)
```

- 显示参数

```python
def show_coeffs(): return dict(zip(indep_cols, coeffs.requires_grad_(False)))
show_coeffs()
```

### 5.5 衡量准确度

```python
preds = calc_preds(coeffs, val_indep)
results = val_dep.bool()==(preds>0.5)
results.float().mean()

#也写成函数
def acc(coeffs): return (val_dep.bool()==(calc_preds(coeffs,val_indep)>0.5)).float().mean()
acc(coeffs)
```

### 5.6 使用激活函数

更新了一下preds的计算函数（参考5.2/5.4）

```python
def calc_preds(coeffs, indeps): return torch.sigmoid((indeps*coeffs).sum(axis=1))
```

其它没什么变化，但为了方便列出

```python
coeffs = train_model(lr=100)
acc(coeffs)
show_coeffs()
```

如果不用激活函数会导致很多数字溢出到0~1之外，最终会特别影响训练效果。比如如果不用激活函数，依然lr=100，那么最后会导致模型不收敛，反之模型收敛得很好，accuracy提高了很多。

### 5.7 测试集

```python
tst_df = pd.read_csv(path/'test.csv')
tst_df['Fare'] = tst_df.Fare.fillna(0)
tst_df.fillna(modes, inplace=True)
tst_df['LogFare'] = np.log(tst_df['Fare']+1)
tst_df = pd.get_dummies(tst_df, columns=["Sex","Pclass","Embarked"])
tst_indep = tensor(tst_df[indep_cols].values, dtype=torch.float)
tst_indep = tst_indep / vals
tst_df['Survived'] = (calc_preds(coeffs, tst_indep)>0.5).int()
sub_df = tst_df[['PassengerId','Survived']]
sub_df.to_csv('sub.csv', index=False)
!head sub.csv
```

### 5.8 matrix转换

还是5.6提到的preds的计算函数，再更新一下，用更加矩阵的方法计算

```python
def calc_preds(coeffs, indeps): return torch.sigmoid(indeps@coeffs)
```

同样的，5.4中coeffs随机生成也要生成矩阵

```python
def init_coeffs(): return (torch.rand(n_coeff, 1)*0.1).requires_grad_()
```

同样的，5.4中训练集和测试集的因变量，也转成矩阵，这个必须要转换成矩阵，这样才能得到n*1的矩阵再求平均，否则在计算loss的时候矩阵-rank1的tensor会被广播成n\*n的矩阵

```python
trn_dep = trn_dep[:,None]
val_dep = val_dep[:,None]
```

### 5.9 神经网络

- 构建模型前参数的形式已经想好了，所以先初始化参数

```python
def init_coeffs(n_hidden=20):
    layer1 = (torch.rand(n_coeff, n_hidden)-0.5)/n_hidden
    layer2 = torch.rand(n_hidden, 1)-0.3
    const = torch.rand(1)[0]
    return layer1.requires_grad_(),layer2.requires_grad_(),const.requires_grad_()
```

- 构建模型的函数

```python
import torch.nn.functional as F
def calc_preds(coeffs, indeps):
    l1,l2,const = coeffs
    res = F.relu(indeps@l1)
    res = res@l2 + const
    return torch.sigmoid(res)
```

- 计算loss的函数

```python
def calc_loss(coeffs, indeps, deps): return torch.abs(calc_preds(coeffs, indeps)-deps).mean()
```

- 更新参数的函数

```python
def update_coeffs(coeffs, lr):
    for layer in coeffs:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()
```

- 1 epoch的函数

```python
def one_epoch(coeffs, lr):
    loss = calc_loss(coeffs, trn_indep, trn_dep) #引用函数
    loss.backward()
    with torch.no_grad(): update_coeffs(coeffs, lr) #引用函数
    print(f"{loss:.3f}", end="; ")
```

- 训练模型的函数

```python
def train_model(epochs=30, lr=0.01):
    torch.manual_seed(442) #为了可复现
    coeffs = init_coeffs()
    for i in range(epocks): one_epoch(coeffs, lr=lr)
    return coeffs
```

- 跑起来

```python
coeffs = train_model(lr=20)
```

- accuracy的函数

```python
def acc(coeffs): return (val_dep.bool()==(calc_preds(coeffs,val_indep)>0.5)).float().mean()
```

- 计算accuracy

```python
acc(coeffs)
```

### 5.10 深度学习

- 数据准备不变
- 构建模型前参数的形式已经想好了，所以先初始化参数

```python
#构建一个多层神经网络，包括2个隐藏层和1个输出层，每个隐藏层的输出都是10个
def init_coeffs():
    hiddens = [10, 10]  # <-- set this to the size of each hidden layer you want
    sizes = [n_coeff] + hiddens + [1]
    n = len(sizes)
    layers = [(torch.rand(sizes[i], sizes[i+1])-0.3)/sizes[i+1]*4 for i in range(n-1)] #元素为tensor的list
    consts = [(torch.rand(1)[0]-0.5)*0.1 for i in range(n-1)] #元素为tensor的list
    for l in layers+consts: l.requires_grad_() #list中的元素简单堆叠
    return layers,consts
```

- 构建模型

```python
import torch.nn.functional as F
def calc_preds(coeffs, indeps):
    layers,consts = coeffs
    n = len(layers)
    res = indeps #输入作为结果的初始化
    for i,l in enumerate(layers):
        res = res@l + consts[i] #这个是tensor的相加会自动广播
        if i!=n-1: res = F.relu(res) #除了输出层，其它层都用relu做激活函数
    return torch.sigmoid(res)
```

- 计算loss的函数（抄5.9）

```python
def calc_loss(coeffs, indeps, deps): return torch.abs(calc_preds(coeffs, indeps)-deps).mean()
```

- 更新参数的函数

```python
def update_coeffs(coeffs, lr):
    layers,consts = coeffs
    for layer in layers+consts:
        layer.sub_(layer.grad * lr)
        layer.grad.zero_()
```

- 1 epoch的函数（抄5.9）

```python
def one_epoch(coeffs, lr):
    loss = calc_loss(coeffs, trn_indep, trn_dep) #引用函数
    loss.backward()
    with torch.no_grad(): update_coeffs(coeffs, lr) #引用函数
    print(f"{loss:.3f}", end="; ")
```

- 训练模型的函数（抄5.9）

```python
def train_model(epochs=30, lr=0.01):
    torch.manual_seed(442) #为了可复现
    coeffs = init_coeffs()
    for i in range(epocks): one_epoch(coeffs, lr=lr)
    return coeffs
```

- 跑起来（抄5.9，但lr做了调整，因为参数多了）

```python
coeffs = train_model(lr=2)
```

- accuracy的函数（抄5.9）

```python
def acc(coeffs): return (val_dep.bool()==(calc_preds(coeffs,val_indep)>0.5)).float().mean()
```

- 计算accuracy（抄5.9）

```python
acc(coeffs)
```

### 5.11 用framework构建深度学习

#### 5.11.1 数据准备

- 与5.1内容对比

```python
#数据下载
from pathlib import Path
import os

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
if iskaggle:
    path = Path('../input/titanic')
    !pip install -Uqq fastai
else:
    import zipfile,kaggle
    path = Path('titanic')
    if not path.exists():
        kaggle.api.competition_download_cli(str(path))
        zipfile.ZipFile(f'{path}.zip').extractall(path)
        
#设置格式
from fastai.tabular.all import * #隐式导入了pandas as pd
pd.options.display.float_format = '{:.2f}'.format
set_seed(42)

#数据处理
df = pd.read_csv(path/'train.csv')
def add_features(df):
    df['LogFare'] = np.log1p(df['Fare'])
    df['Deck'] = df.Cabin.str[0].map(dict(A="ABC", B="ABC", C="ABC", D="DE", E="DE", F="FG", G="FG"))
    df['Family'] = df.SibSp+df.Parch
    df['Alone'] = df.Family==0
    df['TicketFreq'] = df.groupby('Ticket')['Ticket'].transform('count')
    df['Title'] = df.Name.str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    df['Title'] = df.Title.map(dict(Mr="Mr",Miss="Miss",Mrs="Mrs",Master="Master"))
add_features(df)
#'Age', 'SibSp', 'Parch', 'LogFare'，'Sex_male', 'Sex_female', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S'
#生成DataLoaders，里面包含了训练集和验证集
splits = RandomSplitter(seed=42)(df)
dls = TabularPandas(
    df, splits=splits,
    procs = [Categorify, FillMissing, Normalize],
    cat_names=["Sex","Pclass","Embarked","Deck", "Title"],
    cont_names=['Age', 'SibSp', 'Parch', 'LogFare', 'Alone', 'TicketFreq', 'Family'],
    y_names="Survived", y_block = CategoryBlock(),
).dataloaders(path=".")
```

#### 5.11.2 模型训练

- 用了现成的全连接模型

```python
#隐藏层是两个，一个是输入自变量个数*10，一个是10*10，输出层是10*1
learn = tabular_learner(dls, metrics=accuracy, layers=[10,10])
learn.lr_find(suggest_funcs=(slide, valley))
#Jeremy说lr选取slide和valley之间的数会很好
```

![](D:\Git\a\Path-Records\img\05-1-1.png)

```python
learn.fit(16, lr=0.03)
```

#### 5.11.3 测试集

- 训练好的模型用全新的测试集去测试泛化能力

```python
#train文件中的Fare是没有空的，但是test中的有空，所以得先补上，再做和train中同样的操作
tst_df = pd.read_csv(path/'test.csv')
tst_df['Fare'] = tst_df.Fare.fillna(0)
add_features(tst_df)

tst_dl = learn.dls.test_dl(tst_df)
preds,_ = learn.get_preds(dl=tst_dl)

tst_dl = learn.dls.test_dl(tst_df)
preds,_ = learn.get_preds(dl=tst_dl)
tst_df['Survived'] = (preds[:,1]>0.5).int()
sub_df = tst_df[['PassengerId','Survived']]
sub_df.to_csv('sub.csv', index=False)
!head sub.csv
```

#### 5.11.4 整体运行多次

- 由于每次参数都是随机生成的，所以随机生成参数的质量也会影响模型最终的训练效果，因此我们多次生成参数训练模型，取平均值

```python
def ensemble():
    learn = tabular_learner(dls, metrics=accuracy, layers=[10,10])
    with learn.no_bar(),learn.no_logging(): learn.fit(16, lr=0.03)
    return learn.get_preds(dl=tst_dl)[0] #[0]是preds，是一个n*2的tensor，[1]是targets
learns = [ensemble() for _ in range(5)] #生成一个list，这个list中包含5个tensor
ens_preds = torch.stack(learns).mean(0) #torch.stack(learns).shape = [5, 418, 2]，ens_preds.shape = [418, 2]
```

- 求平均值有好几种方法：①直接求5次概率的平均值，再01化；②先01化再求5次平均值，再01化；③先01化然后取众数。一般情况下①和②的效果相对比较好，偶尔③比较好。

```python
tst_df['Survived'] = (ens_preds[:,1]>0.5).int()
sub_df = tst_df[['PassengerId','Survived']]
sub_df.to_csv('ens_sub.csv', index=False)
```

## 5: Tabular (fastbook9)

### 5.1 分类自变量的处理——分类嵌入

- 嵌入层Embedding = 独热编码one-hot + 线性层（没有bias）

- Embedding本身是一个rank-2的矩阵

- 举例：

  - 独热编码+线性层

    假设你有 5 个类别，想要将其映射成一个长度为 3 的向量。你可以这样做：①独热编码（one-hot）一个类别：

    ```
    类别 2 → [0, 1, 0, 0, 0]
    ```

    ②然后通过线性层（权重矩阵 `W.shape = (5, 3)`）：

    ```
    输出 = one_hot_vector @ W
    ```

    这个本质就是用 one-hot 选中矩阵 `W` 中的某一行。
  
  - 嵌入层
  
    嵌入层也是一个查表操作，`Embedding(num_embeddings=5, embedding_dim=3)` 本质上维护一个形状为 `(5, 3)` 的权重矩阵，每次根据类别索引，直接取出对应行作为输出。所以它做的事情是：
  
    ```
    embedding = nn.Embedding(5, 3)
    output = embedding(torch.tensor([2]))  # 就是返回 embedding.weight[2]
    ```

![](D:\Git\a\Path-Records\img\dlcf_0901.png)

- 这本书举例了一篇论文，论文是通过各种数据预测销量，其中一个自变量就是城市，训练好之后作者画出了嵌入矩阵（下图），可以看到embedding自动学习到了城市的地理位置。事实上，在训练之后，商店嵌入之间的距离和实际的地理位置距离非常相近。

![](D:\Git\a\Path-Records\img\dlcf_0902.png)

- 独热dummy和嵌入的差异：类别多、使用深度学习 → 强烈建议使用嵌入（embedding）

| 对比维度            | 独热编码（One-hot Encoding）           | 嵌入（Embedding）                                    |
| ------------------- | -------------------------------------- | ---------------------------------------------------- |
| **维度大小**        | 高维稀疏（每个类别一个维度）           | 低维稠密（通常几维~几十维）                          |
| **内存/计算效率**   | 占内存多，计算慢                       | 占内存少，计算快                                     |
| **类别相似性表达**  | 无法表达类别间关系（每个类别彼此独立） | 可以表达类别间相似性（距离越近越相似）               |
| **是否可训练**      | 否，固定不可学习                       | 是，嵌入向量是模型参数，**可学习**                   |
| **模型泛化能力**    | 容易过拟合，特别是类别很多时           | 泛化能力更强，能从相似类别中借力                     |
| **适合的模型类型**  | 线性模型、树模型等传统机器学习模型     | 深度学习模型（尤其是神经网络）                       |
| **可解释性/可视化** | 不具备可视化语义结构的能力             | 嵌入向量可降维可视化（如用 t-SNE、PCA 展示聚类结构） |
| **类别数量适应性**  | 类别少（<10）较适合                    | 类别多（几十个到几千个）更适合                       |
| **训练速度影响**    | 训练慢，尤其是高维数据                 | 训练快，参数少，优化空间小                           |

### 5.2 Beyond Deep Learning

现代机器学习可以归结为几种广泛适用的关键技术。最近的研究表明，绝大多数数据集最适合用两种方法建模：

+   决策树集成（即随机森林和梯度提升机），主要用于结构化数据（比如大多数公司数据库表中可能找到的数据）训练快，解实性强，有工具和方法可以回答相关问题，比如：数据集中哪些列对你的预测最重要？它们与因变量有什么关系？它们如何相互作用？哪些特定特征对某个特定观察最重要？

+   使用 SGD 学习的多层神经网络（即浅层和/或深度学习），主要用于非结构化数据（比如音频、图像和自然语言）

这一准则的例外情况是当数据集符合以下条件之一时：

+   有一些高基数分类变量非常重要（“基数”指代表示类别的离散级别的数量，因此高基数分类变量是指像邮政编码这样可能有数千个可能级别的变量）。

+   有一些包含最好用神经网络理解的数据的列，比如纯文本数据。

在实践中，事情往往没有那么明确，通常会有高基数和低基数分类变量以及连续变量的混合。很明显我们需要将**决策树**集成添加到我们的建模工具箱中！

要用到决策树，我们需要几个package：Scikit-learn 是一个流行的库，用于创建机器学习模型，使用的方法不包括深度学习。此外，我们需要进行一些表格数据处理和查询，因此我们将使用 Pandas 库。最后，我们还需要 NumPy，因为这是 sklearn 和 Pandas 都依赖的主要数值编程库。

### 5.3 数据集

#### 5.3.1 处理数据集

```python
from fastai.tabular.all import *
path = Path('../input/bluebook-for-bulldozers')
path.ls(file_type='text')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv(path/'TrainAndValid.csv', low_memory=False)
df.columns
#一些分类数据可能做的整理
df.ProductSize.unique()
sizes = 'Large','Large / Medium','Medium','Small','Mini','Compact'
df['ProductSize'] = df['ProductSize'].astype('category')
df['ProductSize'] = df['ProductSize'].cat.set_categories(sizes, ordered=True)
df.ProductSize.unique()
#因变量
df['SalePrice']=np.log(df['SalePrice'])

#日期类数据的处理
df = add_datepart(df, 'saledate')
df_test = pd.read_csv(path/'Test.csv', low_memory=False)
df_test = add_datepart(df_test, 'saledate')
' '.join(o for o in df.columns if o.startswith('sale'))
```

- 训练决策树的基本步骤可以很容易地写下来：

  1、依次循环数据集的每一列。

  2、对于每一列，依次循环该列的每个可能级别。

  3、尝试将数据分成两组，基于它们是否大于或小于该值（或者如果它是一个分类变量，则基于它们是否等于或不等于该分类变量的水平）。

  4、找到这两组中每组的平均销售价格，并查看这与该组中每个设备的实际销售价格有多接近。将这视为一个非常简单的“模型”，其中我们的预测只是该项组的平均销售价格。

  5、在循环遍历所有列和每个可能的级别后，选择使用该简单模型给出最佳预测的分割点。

  6、现在我们的数据有两组，基于这个选定的分割。将每个组视为一个单独的数据集，并通过返回到步骤 1 为每个组找到最佳分割。7、递归地继续这个过程，直到每个组达到某个停止标准-例如，当组中只有 20 个项目时停止进一步分割。

- 我们可以节省一些时间，使用内置在 sklearn 中的实现。为此，要做一些数据准备。

#### 5.3.2 处理字符串-使用 TabularPandas 和 TabularProc

- 处理字符串和缺失数据

```python
#拆分测试集和验证集id
cond = (df.saleYear<2011) | (df.saleMonth<10)
train_idx = np.where( cond)[0]
valid_idx = np.where(~cond)[0]
splits = (list(train_idx),list(valid_idx))

#识别一些连续自变量、分类自变量和因变量，并使得它们具有数字是属性
procs = [Categorify, FillMissing]
dep_var = 'SalePrice'
cont,cat = cont_cat_split(df, 1, dep_var=dep_var) #cont返回连续自变量的title
to = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)
#len(to.train),len(to.valid)
#to.show(3) #可以看到to没有saledate，变成了很多日期拆分；saleprice在最后
#to.items.head(3) #变量都不是字符串了，而是数字

#保存
'''
from fastcore.foundation import Path
from fastcore.xtras import save_pickle, load_pickle
path_out = Path('../working')
save_pickle(path_out/'to.pkl', to)
to_loaded = load_pickle(path_out/'to.pkl')
'''
```

- TabularPandas：是 fastai 库中用于处理表格数据（tabular data）的一个关键类，用于封装并预处理表格数据，以便用于深度学习模型，它相当于把 pandas.DataFrame 包装成了一个可直接用于模型训练的结构（）。

| 功能                          | 说明                                                         |
| ----------------------------- | ------------------------------------------------------------ |
| **预处理数据**                | 包括类别编码（Categorify）、缺失值填补（FillMissing）、数值标准化（Normalize）等 |
| **划分数据集**                | 训练集、验证集等                                             |
| **支持 fastai 的 DataLoader** | 可以用 `.dataloaders()` 直接转成可训练的 DataLoader          |
| **封装处理流程**              | 包含 `procs`（预处理流程）、`cont_names`（连续变量）、`cat_names`（分类变量）等参数 |

### 5.4 决策树

决策树通常情况下是二分法的；

而且同一个特征在同一棵树上可能出现在两个或两个以上节点上，尤其是当特征与目标变量关系较强时；

选择节点：①遍历所有特征，②对于每个特征，尝试所有可能的切分点（数值特征尝试阈值、分类特征尝试子集），③计算每个切分方式下的“不纯度指标”（如基尼指数），④选择能够带来最大纯度提升（即最大信息增益或最小基尼指数）的特征和切分点，⑤创建该节点并向下继续分裂（递归）；

剪枝：最大深度（树的层），节点样本数，子节点样本数，节点完全纯净，没有特征再划分，划分不能带来足够增益，最大叶子节点树。

#### 5.4.1 创建决策树

- 试一试决策树

```python
#根据TabularPandas构建自变量和因变量
xs,y = to.train.xs,to.train.y
valid_xs,valid_y = to.valid.xs,to.valid.y

#训练决策树
from sklearn.tree import DecisionTreeRegressor
m = DecisionTreeRegressor(max_leaf_nodes=4)
m.fit(xs, y);

#绘制决策树
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plot_tree(m, feature_names=xs.columns, filled=True, precision=2)
plt.show()
!pip install -U dtreeviz
import dtreeviz
samp_idx = np.random.permutation(len(y))[:500]
viz = dtreeviz.model(
    m,                                # 你的训练好的模型
    X_train=xs.iloc[samp_idx],              # 特征变量
    y_train=y.iloc[samp_idx],               # 目标变量
    feature_names=xs.columns.tolist(),      # 列名
    target_name=dep_var                    # 字符串形式的目标列名，比如 "SalePrice"
)
viz.view(orientation="LR", scale=1.6, label_fontsize=8, fontname="DejaVu Sans")

#修正一些显然的问题，目的是使得图像更加清晰
xs.loc[xs['YearMade']<1900, 'YearMade'] = 1950
valid_xs.loc[valid_xs['YearMade']<1900, 'YearMade'] = 1950
m = DecisionTreeRegressor(max_leaf_nodes=4).fit(xs, y)
viz = dtreeviz.model(
    m,
    X_train=xs.iloc[samp_idx],
    y_train=y.iloc[samp_idx],
    feature_names=xs.columns.tolist(),
    target_name=dep_var
)
viz.view(orientation="LR", scale=1.6, label_fontsize=8, fontname="DejaVu Sans")
```

![](D:\Git\a\Path-Records\img\dlcf_09in02.png)

![](D:\Git\a\Path-Records\img\dlcf_09in03.png)

- 深化决策树

```python
#疯狂分叉，不设置任何停止条件
m = DecisionTreeRegressor()
m.fit(xs, y);
def r_mse(pred,y): return round(math.sqrt(((pred-y)**2).mean()), 6)
def m_rmse(m, xs, y): return r_mse(m.predict(xs), y)
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)
'(0.0, 0.335804)过拟合了'

#查看决策树叶数量
m.get_n_leaves(), len(xs)
'(324555, 404710)叶子节点个数特别多，分叉太过了'

#设置每个叶子上面最少的样本数量
m = DecisionTreeRegressor(min_samples_leaf=25)
m.fit(to.train.xs, to.train.y)
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)
'(0.248564, 0.32344)好一点了'
m.get_n_leaves(), len(xs)
'(12397, 404710)'
```

#### 5.4.2 决策树中的分类变量

- Pandas 有一个`get_dummies`方法可以做到这一点。然而，实际上并没有证据表明这种方法会改善最终结果。因此，我们通常会尽可能避免使用它，因为它确实会使您的数据集更难处理。只要将它当成正常变量一起处理即可。（不过在深度学习时就会用到）

### 5.5 随机森林

- bagging

1.  随机选择数据的子集（即“学习集的自助复制”）。
1.  使用这个子集训练模型。
1.  保存该模型，然后返回到步骤 1 几次。
1.  这将为您提供多个经过训练的模型。要进行预测，请使用所有模型进行预测，然后取每个模型预测的平均值。

#### 5.5.1 创建随机森林

```python
from sklearn.ensemble import RandomForestRegressor
def rf(xs, y, n_estimators=40, max_samples=200_000,
       max_features=0.5, min_samples_leaf=5, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,
        max_samples=max_samples, max_features=max_features,
        min_samples_leaf=min_samples_leaf, oob_score=True).fit(xs, y)
# 并行使用所有 CPU 核心加速训练
# 树的数量
# 每棵树用的样本量
# 每次分裂考虑的最大特征比例
# 最小叶子样本数
# 启用 OOB 估计，用于模型评估
m = rf(xs, y);
m_rmse(m, xs, y), m_rmse(m, valid_xs, valid_y)
'(0.170992, 0.233527)'
```

随机森林最重要的特性之一是它对超参数选择不太敏感

#### 5.5.2 袋外误差

在随机森林中，每棵树的训练数据是从原始训练集有放回采样（bootstrap sampling）得到的，也就是说：

- 每棵树都用的是训练集的一个“有放回抽样”的子集（约63%的原始样本）。
- 剩下未被抽中的约37%样本就叫做“袋外样本”（Out-Of-Bag samples）。

1. 对每一个样本 xix_ixi：
   - 找出那些没有使用它训练的树（也就是它是这些树的袋外样本）。
   - 用这些树对 xix_ixi 进行预测。
   - 取这些预测的平均值（回归）或投票（分类）作为 xix_ixi 的最终预测。
2. 把所有样本的真实值和袋外预测值对比，计算整体误差，作为 **OOB误差估计**。

```python
r_mse(m.oob_prediction_, y)
'0.21091'
```

我们可以看到我们的 OOB 错误远低于验证集错误（0.233527）。这意味着除了正常的泛化错误之外，还有其他原因导致了该错误。

### 5.6 模型解释

对于表格数据，模型解释尤为重要。对于给定的模型，我们最有兴趣的是以下内容：

+   我们对使用特定数据行进行的预测有多自信？——预测置信度

+   对于使用特定数据行进行预测，最重要的因素是什么，它们如何影响该预测？——特征重要性+树解释器

+   哪些列是最强的预测因子，哪些可以忽略？——特征重要性

+   哪些列在预测目的上实际上是多余的？——特征重要性

+   当我们改变这些列时，预测会如何变化？——部分依赖

#### 5.6.1 树方差——预测置信度

使用树之间预测的标准差，而不仅仅是均值，这告诉我们预测的*相对*置信度。一般来说，我们会更谨慎地使用树给出非常不同结果的行的结果（更高的标准差），而不是在树更一致的情况下使用结果（更低的标准差）。如下：

```python
#置信度
preds = np.stack([t.predict(valid_xs) for t in m.estimators_])
print(preds.shape)
preds_std = preds.std(0)
preds_std[:5]
'''
array([0.30618819, 0.13729507, 0.09630049, 0.25064758, 0.12353961])
第一个置信度低，第三个置信度高
'''
```

#### 5.6.2 特征重要性

##### （1）特征贡献度

**❀我们可以直接从 sklearn 的随机森林中获取这些信息，方法是查看`feature_importances_`属性**。如下：

```python
#重要性
def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)
fi = rf_feat_importance(m, xs)
#表格形式
fi[:10]
#绘图形式
def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:30]);
```

![](D:\Git\a\Path-Records\img\dlcf_09in05.png)

特征重要性的计算：

- 基于节点纯度增益（Gini impurity / MSE 减少）（这是 Scikit-learn 中默认的方法）

  如果一个特征在分裂节点时带来了较大的纯度提升（分类任务）或方差减少（回归任务），说明它对模型更重要。

  - 分类任务中：特征重要性 = 该特征在所有节点上带来的纯度提升的总和（加权平均）
  - 回归任务中：特征重要性 = 每次使用该特征分裂时，训练误差（MSE）下降的总和（加权）
  - 加权说明：每次分裂对 impurity 的改善，会根据当前节点的样本数量进行加权（样本多则更重要）。

- 基于“袋外误差增加”计算（Permutation Importance，打乱法）（这个方法不是默认的，但更稳定、更具解释性）
  - 用原始数据训练好随机森林。
  - 记录袋外（OOB）样本的预测准确率。
  - 对某个特征列随机打乱顺序（破坏其与目标的关联）。
  - 再次计算袋外样本的预测准确率。
  - 重要性 = 准确率下降幅度

- SHAP（外部库-包含但不限于树模型如：XGBoost、随机森林RF）

  - 假设只有 3 个特征：A、B、C，我们要计算特征 A 的 SHAP 值，需要做：列举所有包含 A 的排列组合（比如 A, BA, CA, BCA, etc.）；看 A 加入每个组合时，预测值改变了多少，比如：预测(BC) = 100, 预测(ABC) = 130 → A 的边际贡献 = 30；对所有组合的边际贡献求平均，这样你就得到了 A 的 Shapley 值。

  - 贡献大≠预测准，SHAP 衡量的是影响力而不是正确性。它回答的问题是：“如果没有这个特征，这次预测会有多大不同？”

  - 正值 SHAP：该特征将预测结果推高了。

    负值 SHAP：该特征将预测结果压低了。

    绝对值大：代表该特征对当前预测的影响越强。

    绝对值小或0：代表该特征对当前预测影响很弱，甚至几乎没参与。

##### （2）选择重要特征

**❀有了特征的重要性之后，为了避免使用太多特征进行拟合，可以选择比较重要的特征重新构建随机森林**：

```python
to_keep = fi[fi.imp>0.005].cols
len(to_keep) '21'
xs_imp = xs[to_keep]
valid_xs_imp = valid_xs[to_keep]
m = rf(xs_imp,y)
m_rmse(m, xs_imp, y), m_rmse(m, valid_xs_imp, valid_y)
'(0.181079, 0.230649)'
'与用所有特征的结果(0.170992, 0.233527)相比，好像也没有性能下降多少，但是可以减少很多计算量和特征需求'
plot_fi(rf_feat_importance(m, xs_imp));
```

![](D:\Git\a\Path-Records\img\dlcf_09in06.png)

##### （3）识别特征间相似度

**❀去除冗余特征：有一些特征特别相似，比如ProductGroup和ProductGroupDesc，应该去除这些冗余的特征。**

```python
#特征之间相关性
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

def cluster_columns(df):
    corr = df.corr()  # 计算相关矩阵
    d = 1 - corr.abs()  # 计算距离矩阵（越相关距离越小）
    dist_linkage = linkage(squareform(d), method='average')  # 层次聚类
    sns.clustermap(corr, row_linkage=dist_linkage, col_linkage=dist_linkage,
                   cmap='coolwarm', center=0, figsize=(10,10))
    plt.show()
cluster_columns(xs_imp)
```

![](D:\Git\a\Path-Records\img\dlcf_09in07.png)

可以看到，红色的部分就是疯狂相干的特征，那么把这些密切相关的特征。

##### （4）检查模型效果

**❀让我们尝试删除一些这些密切相关特征，看看模型是否可以简化而不影响准确性。**

```python
#查看去除共线特征的效果
def get_oob(df):
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=15,
        max_samples=50000, max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(df, y)
    return m.oob_score_  #对拟合随机森林来说，obb_score_是R²
get_oob(xs_imp) #基线
'0.8761015614269996'
{c:get_oob(xs_imp.drop(c, axis=1)) for c in (
    'saleYear', 'saleElapsed', 'ProductGroupDesc','ProductGroup',
    'fiModelDesc', 'fiBaseModel',
    'Hydraulics_Flow','Grouser_Tracks', 'Coupler_System')}
'''
{'saleYear': 0.8759547835051612,
 'saleElapsed': 0.8731632755222668,
 'ProductGroupDesc': 0.8773314254721,
 'ProductGroup': 0.8774958058240945,
 'fiModelDesc': 0.8755700975380052,
 'fiBaseModel': 0.8757558713831249,
 'Hydraulics_Flow': 0.877585297419066,
 'Grouser_Tracks': 0.8773646824594399,
 'Coupler_System': 0.877924411741774}
'''
to_drop = ['saleYear', 'ProductGroupDesc', 'fiBaseModel', 'Grouser_Tracks']
get_oob(xs_imp.drop(to_drop, axis=1)) #更新
'0.8745177203238274'
xs_final = xs_imp.drop(to_drop, axis=1)
valid_xs_final = valid_xs_imp.drop(to_drop, axis=1)
m = rf(xs_final, y)
m_rmse(m, xs_final, y), m_rmse(m, valid_xs_final, valid_y)
'(0.182724, 0.233131)；在去除共线特征前是(0.180774, 0.230802)；用所有特征的结果(0.170992, 0.233527)'
'''
Index(['YearMade', 'ProductSize', 'Coupler_System', 'fiProductClassDesc',
       'ModelID', 'saleElapsed', 'Hydraulics_Flow', 'fiSecondaryDesc',
       'Enclosure', 'ProductGroup', 'fiModelDesc', 'SalesID', 'MachineID',
       'Hydraulics', 'fiModelDescriptor', 'Drive_System'],
      dtype='object')
'''
```

| 属性            | 是否基于袋外样本 | 用途                                         | 举例                                    |
| --------------- | ---------------- | -------------------------------------------- | --------------------------------------- |
| oob_prediction_ | 是               | 记录每个样本在未被用于训练的树中得到的预测值 | 一个数组，每个元素是一个样本的 OOB 预测 |
| oob_score_      | 是               | 衡量整体模型性能，在袋外样本上的得分         | 通常是回归中的 R² 或分类准确率          |

#### 5.6.3 部分依赖-PDP图

部分依赖图试图回答这个问题：如果一行除了关注的特征之外没有变化，它会如何影响因变量？

已知'YearMade', 'ProductSize'是非常重要的特征，先看一下这两个特征的情况：

```python
p = valid_xs_final['ProductSize'].value_counts(sort=False).plot.barh()
c = to.classes['ProductSize']
plt.yticks(range(len(c)), c);
ax = valid_xs_final['YearMade'].hist()
```

然后绘制偏依赖图PDP图，以YearMade为例，我们将YearMade列中的每个值替换为 1950，然后计算每个拍卖品的预测销售价格，并对所有拍卖品进行平均。然后我们对 1951、1952 等年份做同样的操作，直到我们的最终年份 2011：

```python
from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(m,valid_xs_final,features=['YearMade','ProductSize'],kind='average').figure_.set_size_inches(12,4)
```

ProductSize的部分图有点令人担忧。它显示我们看到的最终组，即缺失值，价格最低。要在实践中使用这一见解，我们需要找出为什么它经常缺失以及这意味着什么。缺失值有时可以是有用的预测因子-这完全取决于导致它们缺失的原因。然而，有时它们可能表明数据泄漏。

- **数据泄露**

关于数据挖掘问题的目标的信息引入，这些信息不应该合法地从中挖掘出来。泄漏的一个微不足道的例子是一个模型将目标本身用作输入，因此得出例如“雨天下雨”的结论。实际上，引入这种非法信息是无意的，并且由数据收集、聚合和准备过程促成。

识别数据泄漏最实用和简单方法，即构建模型，然后执行以下操作：

① 检查模型的准确性是否过于完美。

② 寻找在实践中不合理的重要预测因子。

③ 寻找在实践中不合理的部分依赖图结果。

- **通常先构建模型，然后进行数据清理是一个好主意，而不是反过来。模型可以帮助您识别潜在的数据问题。**

它还可以帮助您确定哪些因素影响特定预测，使用树解释器。

##### （7）树解释器

