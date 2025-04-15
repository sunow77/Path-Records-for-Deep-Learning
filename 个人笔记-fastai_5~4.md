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

### 4: nlp (fastbook10)

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

