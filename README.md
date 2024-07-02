# Alchemy Cat

[![PyPI version](https://badge.fury.io/py/alchemy-cat.svg)](https://badge.fury.io/py/alchemy-cat)

![banner](docs/figs/dl_config_logo.svg)

<p align="center">
  AlchemyCat 为深度学习提供了一套先进的配置系统。<br> 语法<strong>简单优雅</strong>，支持继承、组合、依赖以<strong>最小化配置冗余</strong>，并支持<strong>自动调参</strong>。
</p>

下表对比了 AlchemyCat 和其他配置系统（😡不支持，🤔有限支持，🥳支持）：

| 功能    | argparse | yaml | YACS | mmcv | AlchemyCat |
|-------|----------|------|------|------|------------|
| 可复现   | 😡       | 🥳   | 🥳   | 🥳   | 🥳         |
| IDE跳转 | 😡       | 😡   | 🥳   | 🥳   | 🥳         |
| 继承    | 😡       | 😡   | 🤔   | 🤔   | 🥳         |
| 组合    | 😡       | 😡   | 🤔   | 🤔   | 🥳         |
| 依赖    | 😡       | 😡   | 😡   | 😡   | 🥳         |
| 自动调参  | 😡       | 😡   | 😡   | 😡   | 🥳         |

AlchemyCat 囊括了此前 "SOTA" 配置系统提供的所有功能，且充分考虑了各种特殊情况，稳定性有保障。

AlchemyCat 的独到之处在于：
* 支持继承、组合来复用已有配置，最小化配置冗余。
* 支持配置项间相互依赖，一处修改，处处生效，大大降低修改配置时的心智负担。
* 提供一台自动调参机，只需对配置文件做一点点修改，即可实现自动调参并总结。
* 且采用了更加简单优雅、pythonic 的语法，附带大量开箱即用的实用方法、属性。

如果您已经在使用上表中某个配置系统，迁移到 AlchemyCat 几乎是零成本的。花15分钟阅读下面的文档，并将 AlchemyCat 运用到项目中，从此你的GPU将永无空闲！

# 安装
```bash
pip install alchemy-cat
```

# 简单使用
AlchemyCat 确保一份配置对应唯一一个实验记录，二者间的双射关系保证了实验的可复现性。
```text
config C + algorithm code A ——> reproducible experiment E(C, A)
```
实验目录是自动创建的，且与配置文件有相同的相对路径。路径可以是多级目录，路径中可以有空格、逗号、等号等。这便于分门别类地管理实验。譬如：
```text
.
├── configs
│   ├── MNIST
│   │   ├── resnet18,wd=1e-5@run2
│   │   │   └── cfg.py
│   │   └── vgg,lr=1e-2
│   │       └── cfg.py
│   └── VOC2012
│       └── swin-T,γ=10
│           └── 10 epoch
│               └── cfg.py
└── experiment
    ├── MNIST
    │   ├── resnet18,wd=1e-5@run2
    │   │   └── xxx.log
    │   └── vgg,lr=1e-2
    │       └── xxx.log
    └── VOC2012
        └── swin-T,γ=10
            └── 10 epoch
                └── xxx.log
```
**最佳实践：在`cfg.py`旁边创建一个`__init__.py`（一般IDE会自动创建），并避免路径中含有'.'。遵守该最佳实践有助于 IDE 调试，且能够在`cfg.py`中使用相对导入。**


让我们从一个不完整的例子开始，了解如何书写配置文件并在代码中加载它。我们首先书写[配置文件](alchemy_cat/dl_config/examples/configs/mnist/plain_usage/cfg.py):
```python
# -- [INCOMPLETE] configs/mnist/plain_usage/cfg.py --

from torchvision.datasets import MNIST
from alchemy_cat.dl_config import Config

cfg = Config()

cfg.rand_seed = 0

cfg.dt.cls = MNIST
cfg.dt.ini.root = '/tmp/data'
cfg.dt.ini.train = True

# ... Code Omitted.
```
这里我们首先实例化一个配置类别对象`cfg`，随后通过属性操作`.`来添加配置项。配置项可以是任意 python 对象，包括函数、方法、类。

**最佳实践：我们推荐直接在配置项中指定函数或类，而不是通过字符串或信号量来控制程序的行为。前者在 IDE 中可以直接跳转，阅读和调试都更加方便。**

`Config`是python `dict`类别的子类，上面代码定义了一个**树结构**的嵌套字典：
```text
>>> print(cfg.to_dict())
{'rand_seed': 0,
 'dt': {'cls': <class 'torchvision.datasets.mnist.MNIST'>,
        'ini': {'root': '/tmp/data', 'train': True}}}
```
`cfg` 支持所有字典用法： 
```test
>>> cfg.keys()
dict_keys(['rand_seed', 'dt'])

>>> cfg['dt']['ini']['root']
'/tmp/data'

>>> {**cfg['dt']['ini'], 'download': True}
{'root': '/tmp/data', 'train': True, 'download': True}
```

可以用字典（yaml、json）或字典的子类（YACS、mmcv.Config）来初始化`Config`对象：
```text
>>> Config({'rand_seed': 0, 'dt': {'cls': MNIST, 'ini': {'root': '/tmp/data', 'train': True}}})
{'rand_seed': 0, 'dt': {'cls': <class 'torchvision.datasets.mnist.MNIST'>, 'ini': {'root': '/tmp/data', 'train': True}}}
```

用属性操作`.`来读写`cfg`会更加清晰，譬如下面代码，根据配置文件创建并初始化了`MNIST`数据集：
```text
>>> dataset = cfg.dt.cls(**cfg.dt.ini)
>>> dataset
Dataset MNIST
    Number of datapoints: 60000
    Root location: /tmp/data
    Split: Train
```
若访问不存在的键，会返回一个一次性空字典，在主代码可将其视作`False`：
```text
>>> cfg.not_exit
{}
```
在[主代码](alchemy_cat/dl_config/examples/train.py)中，使用下面代码加载配置：
```python
# # [INCOMPLETE] -- train.py --

from alchemy_cat.dl_config import load_config
cfg = load_config('configs/mnist/base/cfg.py', experiments_root='/tmp/experiment', config_root='configs')
# ... Code Omitted.
torch.save(model.state_dict(), f"{cfg.rslt_dir}/model_{epoch}.pth")  # Save all experiment results to cfg.rslt_dir.
```
`load_config`函数会导入`configs/mnist/base/cfg.py`中的`cfg`，处理继承、依赖。指定实验和配置的根目录`experiments_root`和`config_root`后，`load_config`会自动创建实验目录`experiment/mnist/base/cfg.py`并赋值给`cfg.rslt_dir`，一切实验结果都应当保存到`cfg.rslt_dir`中。

加载得到的`cfg`默认是冻结的，即`cfg.is_frozen == True`，此时不允许增删改配置。若要修改配置，可以通过`cfg.unfreeze()`解冻。


## 本章小结
* 配置文件提供一个`Config`对象`cfg`，其本质是一个树结构的嵌套字典，支持`.`操作读写。
* `cfg`访问不存在的键时，返回一个一次性空字典，可将其视作`False`。
* 使用`load_config`函数加载配置文件，实验目录会自动创建，其路径会赋值给`cfg.rslt_dir`。

# 继承
新配置可以继承已有的基配置，写作`cfg = Config(caps='base_cfg.py')`。如此可以复用基配置，新配置只需覆写要修改的项目，或新增一些项目。如对[基配置](alchemy_cat/dl_config/examples/configs/mnist/plain_usage/cfg.py)：
```python
# -- [INCOMPLETE] configs/mnist/plain_usage/cfg.py --

# ... Code Omitted.

cfg.loader.ini.batch_size = 128
cfg.loader.ini.num_workers = 2

cfg.opt.cls = optim.AdamW
cfg.opt.ini.lr = 0.01

# ... Code Omitted.
```
如果新的实验想翻倍批次大小，[新配置](alchemy_cat/dl_config/examples/configs/mnist/plain_usage,2xbs/cfg.py)可写作：
```python
# -- configs/mnist/plain_usage,2xbs/cfg.py --

from alchemy_cat.dl_config import Config

cfg = Config(caps='configs/mnist/plain_usage/cfg.py')  # Inherit from base config.

cfg.loader.ini.batch_size = 128 * 2  # Double batch size.

cfg.opt.ini.lr = 0.01 * 2  # Linear scaling rule, see https://arxiv.org/abs/1706.02677
```
继承的行为和字典的`update`类似。核心区别在于，当新配置和基配置有同名键，且值也是一棵配置树时（称作“子配置树”），我们会递归进入子配置树执行`update`操作。因此，新配置的`cfg.loader.ini.num_workers`并未丢失，而是依旧保持基配置的值。
```text
>>> base_cfg = load_config('configs/mnist/plain_usage/cfg.py', create_rslt_dir=False)
>>> new_cfg = load_config('configs/mnist/plain_usage,2xbs/cfg.py', create_rslt_dir=False)
>>> base_cfg.loader.ini
{'batch_size': 128, 'num_workers': 2}
>>> new_cfg.loader.ini
{'batch_size': 256, 'num_workers': 2}
```
若想在新配置中重写整棵子配置树，可将其设子树置为 "whole"，[例如](alchemy_cat/dl_config/examples/configs/mnist/plain_usage,override_loader/cfg.py)：
```python
# -- configs/mnist/plain_usage,override_loader/cfg.py --

from alchemy_cat.dl_config import Config

cfg = Config(caps='configs/mnist/plain_usage/cfg.py')  # Inherit from base config.

cfg.loader.ini.set_whole()  # Set subtree as whole.
cfg.loader.ini.shuffle = False
cfg.loader.ini.drop_last = False
```
此时，`cfg.loader.ini`将完全由新配置定义：
```text
>>> base_cfg = load_config('configs/mnist/plain_usage/cfg.py', create_rslt_dir=False)
>>> new_cfg = load_config('configs/mnist/plain_usage,2xbs/cfg.py', create_rslt_dir=False)
>>> base_cfg.loader.ini
{'batch_size': 128, 'num_workers': 2}
>>> new_cfg.loader.ini
{'shuffle': False, 'drop_last': False}
```
自然而然地，让基配置可以可以继承自另一个基配置，可以实现链式继承。

最后，配置也支持多继承，写作`cfg = Config(caps=('base.py', 'patch1.py', 'patch2.py', ...))`，其建立一条`base -> patch1 -> patch2 -> current cfg`的继承链。这种写法中，靠右的基配置常作为补丁项，用于添加一套经常共现的配置项。例如下面的[patch](alchemy_cat/dl_config/examples/configs/patches/cifar10.py):
```python
# -- configs/patches/cifar10.py --

import torchvision.transforms as T
from torchvision.datasets import CIFAR10

from alchemy_cat.dl_config import Config

cfg = Config()

cfg.dt.set_whole()
cfg.dt.cls = CIFAR10
cfg.dt.ini.root = '/tmp/data'
cfg.dt.ini.transform = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
```
当[新配置](alchemy_cat/dl_config/examples/configs/mnist/plain_usage,cifar10/cfg.py)需要改用CIFAR10数据集时，只需要继承该补丁即可，无需再写CIFAR10有关的配置：
```python
# -- configs/mnist/plain_usage,cifar10/cfg.py --

from alchemy_cat.dl_config import Config

cfg = Config(caps=('configs/mnist/plain_usage/cfg.py', 'alchemy_cat/dl_config/examples/configs/patches/cifar10.py'))
```
```text
>>> cfg = load_config('configs/mnist/plain_usage,cifar10/cfg.py', create_rslt_dir=False)
>>> cfg.dt
{'cls': torchvision.datasets.cifar.CIFAR10,
 'ini': {'root': '/tmp/data',
  'transform': Compose(
      ToTensor()
      Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
  )}}
```
> _继承的实现细节_
> 
> 继承时，我们先拷贝整棵基配置树，再以新配置更新之，确保新配置和基配置的树结构相互隔离——即修改新配置的结构不会影响基配置。故而更复杂继承关系，如菱形继承也是支持的，只是不太可读，不推荐使用。\
> 同时注意，叶结点的值是引用传递，原地修改将在影响整条继承链。

## 本章小结
* 新配置可以继承已有的基配置，复用基配置的项目，并覆写、新增一些配置项。
* 新配置对基配置的更新是递归进行的，使用`Config.set_whole`可以退回到`dict`默认的更新方式。
* `Config`支持链式继承和多继承，多继承可实现更加细粒度的复用。

# 依赖
[上一节](#继承)的例子中，当新配置修改基配置的批次大小时，学习率也随之变化。这种一个配置项随着另一个配置项的变化情况，称作“依赖”。

在修改某个配置项的时候，忘记修改它的依赖项，是非常常见的 bug。 好在 AlchemyCat 可以定义依赖项，如此，每次只需要修改依赖的源头，所有依赖项都会自动更新。[例如](alchemy_cat/dl_config/examples/configs/mnist/base/cfg.py)：
```python
# -- [INCOMPLETE] configs/mnist/base/cfg.py --

from alchemy_cat.dl_config import Config, DEP
# ... Code Omitted.

cfg.loader.ini.batch_size = 128
# ... Code Omitted.
cfg.opt.ini.lr = DEP(lambda c: c.loader.ini.batch_size // 128 * 0.01)  # Linear scaling rule.

# ... Code Omitted.
```
其中，学习率`cfg.opt.ini.lr`作为依赖项`DEP`，借助批次大小`cfg.loader.ini.batch_size`算出。`DEP`接受一个函数，函数的实参将会是`cfg`，并返回依赖项的值。

在[新配置](alchemy_cat/dl_config/examples/configs/mnist/base,2xbs/cfg.py)中，我们只需修改批次大小，学习率会自动更新：
```python
# -- configs/mnist/base,2xbs/cfg.py --

from alchemy_cat.dl_config import Config

cfg = Config(caps='configs/mnist/base/cfg.py')

cfg.loader.ini.batch_size = 128 * 2  # Double batch size, learning rate will be doubled automatically.
```
```text
>>> cfg = load_config('configs/mnist/base,2xbs/cfg.py', create_rslt_dir=False)
>>> cfg.loader.ini.batch_size
256
>>> cfg.opt.ini.lr
0.02
```
下面展示一个更复杂的[例子](alchemy_cat/dl_config/examples/configs/mnist/base/cfg.py)：
```python
# -- configs/mnist/base/cfg.py --

# ... Code Omitted.

cfg.sched.epochs = 30
@cfg.sched.set_DEP(name='warm_epochs', priority=0)  # kwarg `name` is not necessary
def warm_epochs(c: Config) -> int:  # warm_epochs = 10% of total epochs
    return round(0.1 * c.sched.epochs)

cfg.sched.warm.cls = sched.LinearLR
cfg.sched.warm.ini.total_iters = DEP(lambda c: c.sched.warm_epochs, priority=1)
cfg.sched.warm.ini.start_factor = 1e-5
cfg.sched.warm.ini.end_factor = 1.

cfg.sched.main.cls = sched.CosineAnnealingLR
cfg.sched.main.ini.T_max = DEP(lambda c: c.sched.epochs - c.sched.warm.ini.total_iters,
                               priority=2)  # main_epochs = total_epochs - warm_epochs

# ... Code Omitted.
```
```text
>>> print(cfg.sched.to_txt(prefix='cfg.sched.'))  # A pretty print of the config tree.
cfg.sched = Config()
# ------- ↓ LEAVES ↓ ------- #
cfg.sched.epochs = 30
cfg.sched.warm_epochs = 3
cfg.sched.warm.cls = <class 'torch.optim.lr_scheduler.LinearLR'>
cfg.sched.warm.ini.total_iters = 3
cfg.sched.warm.ini.start_factor = 1e-05
cfg.sched.warm.ini.end_factor = 1.0
cfg.sched.main.cls = <class 'torch.optim.lr_scheduler.CosineAnnealingLR'>
cfg.sched.main.ini.T_max = 27
```
上面代码中，总的训练轮次`cfg.sched.epochs`是依赖源头，预热轮次 `cfg.sched.warm_epochs`是总训练轮次的 10%，主轮次`cfg.sched.main.ini.T_max`是总训练轮次减去预热轮次。只需修改总训练轮次，依赖项预热轮次和主轮次都会自动更新。

依赖项`cfg.sched.warm_epochs`使用了`Config.set_DEP`装饰器来定义，所饰函数即`DEP`的首个参数，定义依赖项的计算方式。装饰器可通过关键字参数`name`指定依赖项的键名，若缺省则使用被饰函数之名。当依赖项的计算函数较为复杂时，推荐使用装饰器来定义。

当依赖项依赖另一个依赖项时，需要保证依赖性按正确顺序解算。默认的解算顺序是定义顺序。也可以通过`priority`参数来指定解算顺序，数值越小，解算越早。譬如上面`cfg.sched.warm_epochs`被`cfg.sched.warm.ini.total_iters`，后者又被`cfg.sched.main.ini.T_max`依赖，故他们的`priority`依次增加。

## 本章小结
* 当一个配置项依赖于另一个配置项时，可将其定义依赖项。改变依赖源头时，依赖项会根据计算函数自动解算，而无需手动修改。
* 依赖项有两种定义方式：直接赋值为`DEP(...)`，或使用`Config.set_DEP`装饰器。
* 依赖项间相互依赖时，可通过`priority`参数来指定解算顺序，否则按照定义顺序解算。

# 组合
组合是另一种复用配置的方式。预定义好的子配置树，可以像积木一样，组合出完整的配置。譬如，下面[配置](alchemy_cat/dl_config/examples/configs/addons/linear_warm_cos_sched.py)定义了一套学习率策略：

```python
# -- configs/addons/linear_warm_cos_sched.py --
import torch.optim.lr_scheduler as sched

from alchemy_cat.dl_config import Config, DEP

cfg = Config()

cfg.epochs = 30

@cfg.set_DEP(priority=0)  # warm_epochs = 10% of total epochs
def warm_epochs(c: Config) -> int:
    return round(0.1 * c.epochs)

cfg.warm.cls = sched.LinearLR
cfg.warm.ini.total_iters = DEP(lambda c: c.warm_epochs, priority=1)
cfg.warm.ini.start_factor = 1e-5
cfg.warm.ini.end_factor = 1.

cfg.main.cls = sched.CosineAnnealingLR
cfg.main.ini.T_max = DEP(lambda c: c.epochs - c.warm.ini.total_iters,
                         priority=2)  # main_epochs = total_epochs - warm_epochs

```
在[最终配置](alchemy_cat/dl_config/examples/configs/mnist/base,sched_from_addon/cfg.py)中，我们直接组合这套学习率策略：
```python
# -- configs/mnist/base,sched_from_addon/cfg.py --
# ... Code Omitted.

cfg.sched = Config('configs/addons/linear_warm_cos_sched.py')

# ... Code Omitted.
```
```text
>>> print(cfg.sched.to_txt(prefix='cfg.sched.'))  # A pretty print of the config tree.
cfg.sched = Config()
# ------- ↓ LEAVES ↓ ------- #
cfg.sched.epochs = 30
cfg.sched.warm_epochs = 3
cfg.sched.warm.cls = <class 'torch.optim.lr_scheduler.LinearLR'>
cfg.sched.warm.ini.total_iters = 3
cfg.sched.warm.ini.start_factor = 1e-05
cfg.sched.warm.ini.end_factor = 1.0
cfg.sched.main.cls = <class 'torch.optim.lr_scheduler.CosineAnnealingLR'>
cfg.sched.main.ini.T_max = 27
```
看起来非常简单！就是将预定义的子配置树，赋值/挂载到最终配置。`Config('path/to/cfg.py')`返回配置文件中，`cfg`对象的拷贝（与[继承](#继承)中一样，拷贝树结构以保证拷贝前后对配置的修改相互隔离）。

> _组合和依赖的实现细节_
>
> 细心的读者可能会疑惑，`DEP`是如何决定依赖项计算函数的参数`c`，解算时具体传入哪一个`Config`对象？在本章的例子中，`c`的实参是学习率子配置，因此`cfg.warm.ini.total_iters`的计算函数为`lambda c: c.warm_epochs`。然而，在[上一章](#依赖)的例子中，`c`的实参是最终配置，因此`cfg.sched.warm.ini.total_iters`的计算函数为`lambda c: c.sched.warm_epochs`。
> 
>其实，`c`的实参，是`DEP`第一次被挂载到配置树时，被挂载的那颗配置树之根节点。`Config`在数据结构上是一棵双向树，`DEP`第一次被挂载时，会上溯到根节点，记录`DEP`到根的相对距离。解算时，上溯相同距离，找到对应的配置树，并传入计算函数。
> 
> 要阻止该默认行为，可以设置`DEP(lambda c: ..., rel=False)`，此时`c`的实参总是为最终配置。
> 


**最佳实践：与面向对象的组合和继承类似，配置的组合和继承，初衷都是为了复用配置代码。其中，组合更加灵活、低耦合。因此，应当优先使用组合，尽量减少继承层次。**

## 本章小结
* 可以先定义几组子配置，再将他们赋值为最终配置的键值对，以此组合出完整的配置。

# 完整样例


<details>
<summary> 展开完整样例 </summary>

学习率相关的[子配置树](alchemy_cat/dl_config/examples/configs/addons/linear_warm_cos_sched.py)：
```python
# -- configs/addons/linear_warm_cos_sched.py --

import torch.optim.lr_scheduler as sched

from alchemy_cat.dl_config import Config, DEP

cfg = Config()

cfg.epochs = 30

@cfg.set_DEP(priority=0)  # warm_epochs = 10% of total epochs
def warm_epochs(c: Config) -> int:
    return round(0.1 * c.epochs)

cfg.warm.cls = sched.LinearLR
cfg.warm.ini.total_iters = DEP(lambda c: c.warm_epochs, priority=1)
cfg.warm.ini.start_factor = 1e-5
cfg.warm.ini.end_factor = 1.

cfg.main.cls = sched.CosineAnnealingLR
cfg.main.ini.T_max = DEP(lambda c: c.epochs - c.warm.ini.total_iters,
                         priority=2)  # main_epochs = total_epochs - warm_epochs
```
组合得到的[基配置](alchemy_cat/dl_config/examples/configs/mnist/base,sched_from_addon/cfg.py)：
```python
# -- configs/mnist/base/cfg.py --

import torchvision.models as model
import torchvision.transforms as T
from torch import optim
from torchvision.datasets import MNIST

from alchemy_cat.dl_config import Config, DEP

cfg = Config()

cfg.rand_seed = 0

# -* Set datasets.
cfg.dt.cls = MNIST
cfg.dt.ini.root = '/tmp/data'
cfg.dt.ini.transform = T.Compose([T.Grayscale(3), T.ToTensor(), T.Normalize((0.1307,), (0.3081,)),])

# -* Set data loader.
cfg.loader.ini.batch_size = 128
cfg.loader.ini.num_workers = 2

# -* Set model.
cfg.model.cls = model.resnet18
cfg.model.ini.num_classes = DEP(lambda c: len(c.dt.cls.classes))

# -* Set optimizer.
cfg.opt.cls = optim.AdamW
cfg.opt.ini.lr = DEP(lambda c: c.loader.ini.batch_size // 128 * 0.01)  # Linear scaling rule.

# -* Set scheduler.
cfg.sched = Config('configs/addons/linear_warm_cos_sched.py')

# -* Set logger.
cfg.log.save_interval = DEP(lambda c: c.sched.epochs // 5, priority=1)  # Save model at every 20% of total epochs.
```
继承自基配置，批次大小翻倍，轮次数目减半的[新配置](alchemy_cat/dl_config/examples/configs/mnist/base,sched_from_addon,2xbs,2÷epo/cfg.py)：

```python
# -- configs/mnist/base,sched_from_addon,2xbs,2÷epo/cfg.py --

from alchemy_cat.dl_config import Config

cfg = Config(caps='configs/mnist/base,sched_from_addon/cfg.py')

cfg.loader.ini.batch_size = 256

cfg.sched.epochs = 15
```
注意，依赖项如学习率、预热轮次、主轮次都会自动更新：
```text
>>> cfg = load_config('configs/mnist/base,sched_from_addon,2xbs,2÷epo/cfg.py', create_rslt_dir=False)
>>> print(cfg)
cfg = Config()
cfg.set_whole(False).set_attribute('_cfgs_update_at_parser', ('configs/mnist/base,sched_from_addon/cfg.py',))
# ------- ↓ LEAVES ↓ ------- #
cfg.rand_seed = 0
cfg.dt.cls = <class 'torchvision.datasets.mnist.MNIST'>
cfg.dt.ini.root = '/tmp/data'
cfg.dt.ini.transform = Compose(
    Grayscale(num_output_channels=3)
    ToTensor()
    Normalize(mean=(0.1307,), std=(0.3081,))
)
cfg.loader.ini.batch_size = 256
cfg.loader.ini.num_workers = 2
cfg.model.cls = <function resnet18 at 0x7f5bcda68a40>
cfg.model.ini.num_classes = 10
cfg.opt.cls = <class 'torch.optim.adamw.AdamW'>
cfg.opt.ini.lr = 0.02
cfg.sched.epochs = 15
cfg.sched.warm_epochs = 2
cfg.sched.warm.cls = <class 'torch.optim.lr_scheduler.LinearLR'>
cfg.sched.warm.ini.total_iters = 2
cfg.sched.warm.ini.start_factor = 1e-05
cfg.sched.warm.ini.end_factor = 1.0
cfg.sched.main.cls = <class 'torch.optim.lr_scheduler.CosineAnnealingLR'>
cfg.sched.main.ini.T_max = 13
cfg.log.save_interval = 3
cfg.rslt_dir = 'mnist/base,sched_from_addon,2xbs,2÷epo'
```
[训练代码](alchemy_cat/dl_config/examples/train.py)：
```python
# -- train.py --
import argparse
import json

import torch
import torch.nn.functional as F
from rich.progress import track
from torch.optim.lr_scheduler import SequentialLR

from alchemy_cat.dl_config import load_config
from utils import eval_model

parser = argparse.ArgumentParser(description='AlchemyCat MNIST Example')
parser.add_argument('-c', '--config', type=str, default='configs/mnist/base,sched_from_addon,2xbs,2÷epo/cfg.py')
args = parser.parse_args()

# Folder 'experiment/mnist/base' will be auto created by `load` and assigned to `cfg.rslt_dir`
cfg = load_config(args.config, experiments_root='/tmp/experiment', config_root='configs')
print(cfg)

torch.manual_seed(cfg.rand_seed)  # Use `cfg` to set random seed

dataset = cfg.dt.cls(**cfg.dt.ini)  # Use `cfg` to set dataset type and its initial parameters

# Use `cfg` to set changeable parameters of loader,
# other fixed parameter like `shuffle` is set in main code
loader = torch.utils.data.DataLoader(dataset, shuffle=True, **cfg.loader.ini)

model = cfg.model.cls(**cfg.model.ini).train().to('cuda')  # Use `cfg` to set model

# Use `cfg` to set optimizer, and get `model.parameters()` in run time
opt = cfg.opt.cls(model.parameters(), **cfg.opt.ini, weight_decay=0.)

# Use `cfg` to set warm and main scheduler, and `SequentialLR` to combine them
warm_sched = cfg.sched.warm.cls(opt, **cfg.sched.warm.ini)
main_sched = cfg.sched.main.cls(opt, **cfg.sched.main.ini)
sched = SequentialLR(opt, [warm_sched, main_sched], [cfg.sched.warm_epochs])

for epoch in range(1, cfg.sched.epochs + 1):  # train `cfg.sched.epochs` epochs
    for data, target in track(loader, description=f"Epoch {epoch}/{cfg.sched.epochs}"):
        F.cross_entropy(model(data.to('cuda')), target.to('cuda')).backward()
        opt.step()
        opt.zero_grad()

    sched.step()

    # If cfg.log is defined, save model to `cfg.rslt_dir` at every `cfg.log.save_interval`
    if cfg.log and epoch % cfg.log.save_interval == 0:
        torch.save(model.state_dict(), f"{cfg.rslt_dir}/model_{epoch}.pth")

    eval_model(model)

if cfg.log:
    eval_ret = eval_model(model)
    with open(f"{cfg.rslt_dir}/eval.json", 'w') as json_f:
        json.dump(eval_ret, json_f)
```

运行`python train.py --config 'configs/mnist/base,sched_from_addon,2xbs,2÷epo/cfg.py'`，将会按照配置文件中设置，使用`train.py`训练，并将结果保存到`/tmp/experiment/mnist/base,sched_from_addon,2xbs,2÷epo`目录中。
</details>

# 自动调参
在[上面的例子](#完整样例)中，我们每运行`python train.py --config path/to/cfg.py`，就针对一组参数，得到一份对应的实验结果。

然而，很多时候，我们需要网格搜索参数空间，寻找最优的参数组合。若为每一组参数组合都写一个配置，既辛苦也容易出错。那能不能在一个『可调配置』中，定义整个参数空间。随后让程序自动地遍历所有参数组合，对每组参数生成一个配置并运行。进一步的，程序还应该能够自动汇总每组实验结果，以便比较。

自动调参机遍历可调配置的参数组合，生成`N`个子配置，运行得到`N`个实验记录，并将所有实验结果总结到 excel 表格中：
```text
config to be tuned T ───> config C1 + algorithm code A ───> reproducible experiment E1(C1, A) ───> summary table S(T,A)
                     │                                                                          │  
                     ├──> config C2 + algorithm code A ───> reproducible experiment E1(C2, A) ──│ 
                    ...                                                                         ...
```
## 可调配置
要使用自动调参机，首先需要写一个可调配置：
```python
# -- configs/tune/tune_bs_epoch/cfg.py --

from alchemy_cat.dl_config import Cfg2Tune, Param2Tune

cfg = Cfg2Tune(caps='configs/mnist/base,sched_from_addon/cfg.py')

cfg.loader.ini.batch_size = Param2Tune([128, 256, 512])

cfg.sched.epochs = Param2Tune([5, 15])
```
其写法与上一章中的[普通配置](alchemy_cat/dl_config/examples/configs/mnist/base,sched_from_addon,2xbs,2÷epo/cfg.py)非常类似，同样支持属性读写、继承、依赖、组合等特性。区别在于：
* 可调配置的的数据类型是`Config`的子类`Cfg2Tune`。
* 对需要网格搜索的参数，定义为`Param2Tune([v1, v2, ...])`，其中`v1, v2, ...`是参数的可选值。

譬如上面的可调配置，会搜索一个 3×2=6 大小的参数空间，并生成如下6个子配置：
```text
batch_size  epochs  child_configs            
128         5       configs/tune/tune_bs_epoch/batch_size=128,epochs=5/cfg.pkl
            15      configs/tune/tune_bs_epoch/batch_size=128,epochs=15/cfg.pkl
256         5       configs/tune/tune_bs_epoch/batch_size=256,epochs=5/cfg.pkl
            15      configs/tune/tune_bs_epoch/batch_size=256,epochs=15/cfg.pkl
512         5       configs/tune/tune_bs_epoch/batch_size=512,epochs=5/cfg.pkl
            15      configs/tune/tune_bs_epoch/batch_size=512,epochs=15/cfg.pkl
```

设置`Param2Tune`的`priority`参数，可以指定参数的搜索顺序。默认搜索顺序是定义顺序。设置` optional_value_names`参数，可以为参数值指定可读的名字。[例如](alchemy_cat/dl_config/examples/configs/tune/tune_bs_epoch,pri,name/cfg.py)：
```python
# -- configs/tune/tune_bs_epoch,pri,name/cfg.py --

from alchemy_cat.dl_config import Cfg2Tune, Param2Tune

cfg = Cfg2Tune(caps='configs/mnist/base,sched_from_addon/cfg.py')

cfg.loader.ini.batch_size = Param2Tune([128, 256, 512], optional_value_names=['1xbs', '2xbs', '4xbs'], priority=1)

cfg.sched.epochs = Param2Tune([5, 15], priority=0)
```
其搜索空间为：
```text
epochs batch_size  child_configs                    
5      1xbs        configs/tune/tune_bs_epoch,pri,name/epochs=5,batch_size=1xbs/cfg.pkl
       2xbs        configs/tune/tune_bs_epoch,pri,name/epochs=5,batch_size=2xbs/cfg.pkl
       4xbs        configs/tune/tune_bs_epoch,pri,name/epochs=5,batch_size=4xbs/cfg.pkl
15     1xbs        configs/tune/tune_bs_epoch,pri,name/epochs=15,batch_size=1xbs/cfg.pkl
       2xbs        configs/tune/tune_bs_epoch,pri,name/epochs=15,batch_size=2xbs/cfg.pkl
       4xbs        configs/tune/tune_bs_epoch,pri,name/epochs=15,batch_size=4xbs/cfg.pk
```

我们还可以在参数间设置约束条件，裁剪掉不需要的参数组合，如下面[例子](alchemy_cat/dl_config/examples/configs/tune/tune_bs_epoch,subject_to/cfg.py)约束总迭代数不超过 15×128：
```python
# -- configs/tune/tune_bs_epoch,subject_to/cfg.py --

from alchemy_cat.dl_config import Cfg2Tune, Param2Tune

cfg = Cfg2Tune(caps='configs/mnist/base,sched_from_addon/cfg.py')

cfg.loader.ini.batch_size = Param2Tune([128, 256, 512])

cfg.sched.epochs = Param2Tune([5, 15],
                              subject_to=lambda cur_val: cur_val * cfg.loader.ini.batch_size.cur_val <= 15 * 128)
```
其搜索空间为：
```text
batch_size epochs  child_configs                 
128        5       configs/tune/tune_bs_epoch,subject_to/batch_size=128,epochs=5/cfg.pkl  
           15      configs/tune/tune_bs_epoch,subject_to/batch_size=128,epochs=15/cfg.pkl
256        5       configs/tune/tune_bs_epoch,subject_to/batch_size=256,epochs=5/cfg.pkl
```

## 运行自动调参机
我们还需要写一小段脚本来运行自动调参机：
```python
# -- tune_train.py --
import argparse, json, os, subprocess, torch, sys
from alchemy_cat.dl_config import Config, Cfg2TuneRunner

parser = argparse.ArgumentParser(description='Tuning AlchemyCat MNIST Example')
parser.add_argument('-c', '--cfg2tune', type=str)
args = parser.parse_args()

# Set `pool_size` to GPU num, will run `pool_size` of configs in parallel
runner = Cfg2TuneRunner(args.cfg2tune, experiment_root='/tmp/experiment', pool_size=torch.cuda.device_count())

@runner.register_work_fn  # How to run config
def work(pkl_idx: int, cfg: Config, cfg_pkl: str, cfg_rslt_dir: str) -> ...:
    subprocess.run([sys.executable, 'train.py', '-c', cfg_pkl],
                   env=os.environ | {'CUDA_VISIBLE_DEVICE': f'pkl_idx % torch.cuda.device_count()'})

@runner.register_gather_metric_fn  # How to gather metric for summary
def gather_metric(cfg: Config, cfg_rslt_dir: str, run_rslt: ..., param_comb: dict[str, tuple[..., str]]) -> dict[str, ...]:
    return json.load(open(os.path.join(cfg_rslt_dir, 'eval.json')))

runner.tuning()
```
上面的脚本执行了如下操作：
* 传入可调配置的路径，实例化调参机`runner = Cfg2TuneRunner(...)`。<br> 
自动调参机默认逐个运行子配置。设置参数`pool_size > 0`，可以并行运行`pool_size`个子配置。对深度学习任务，`pool_size`一般为`GPU数量 // 每个任务所占GPU数量`。
* 注册工作函数。子配置将被调参机逐个传入工作函数并运行之。<br>
工作函数接受如下参数：`pkl_idx`是子配置的序号；`cfg`是子配置；`cfg_pkl`是子配置的 pickle 保存路径；`cfg_rslt_dir`是子配置的实验目录。一般而言，我们只需要将`cfg_pkl`作为配置文件（`load_cfg`支持读取 pickle 保存的配置）传入训练脚本即可。对深度学习任务，如上例所示，还需要为每个任务设置不同的`CUDA_VISIBLE_DEVICE`。
* 注册汇总函数。汇总函数对每个实验结果，返回一个字典，格式为`{metric_name: metric_value}`。调参机会自动遍历所有实验结果，汇总到一个表格中。<br>
汇总函数接受如下参数：`cfg`是子配置；`cfg_rslt_dir`是子配置的实验目录；`run_rslt`是工作函数的返回值；`param_comb`是子配置的参数组合。一般我们只需要到`cfg_rslt_dir`中读取实验结果并返回即可。
* 调用`runner.tuning()`，开始自动调参。

调参结束后，将打印调参结果：
```text
Metric Frame: 
                  test_loss    acc
batch_size epochs                 
128        5       1.993285  32.63
           15      0.016772  99.48
256        5       1.889874  37.11
           15      0.020811  99.49
512        5       1.790593  41.74
           15      0.024695  99.33

Saving Metric Frame at /tmp/experiment/tune/tune_bs_epoch/metric_frame.xlsx
```
正如提示信息所言，调参结果还会被保存到 `/tmp/experiment/tune/tune_bs_epoch/metric_frame.xlsx` 表格中：
![metric_frame](docs/figs/readme-cfg2tune-excel.png)

**最佳实践：自动调参机与标准的工作流是正交的。因此，在写配置和代码时，先不要考虑自动调参机。需要调参时，再写一点点额外的代码，定义参数空间，指定算法的调用方式和结果的获取方式。调参完毕后，可以剥离调参机，只发布最优结果的配置和算法。**

## 本章小结
* 我们可以在可调配置`Cfg2Tune`下，使用`Param2Tune`定义参数空间。
* 自动调参机`Cfg2TuneRunner`会遍历参数空间，生成子配置，运行子配置，并汇总实验结果。

# 进阶

<details>
<summary> 展开进阶 </summary>

## 美化打印
`Config`的`__str__`方法被重载，以`.`分隔的键名，美观地打印树结构：

```text
>>> cfg = Config()
>>> cfg.foo.bar.a = 1
>>> cfg.bar.foo.b = ['str1', 'str2']
>>> cfg.whole.set_whole()
>>> print(cfg)
cfg = Config()
cfg.whole.set_whole(True)
# ------- ↓ LEAVES ↓ ------- #
cfg.foo.bar.a = 1
cfg.bar.foo.b = ['str1', 'str2']
```

如果所有叶节点都是内置类型，`Config`的美观打印输出可直接作为 python 代码执行，并得到相同的配置：
```text
>>> exec(cfg.to_txt(prefix='new_cfg.'), globals(), (l_dict := {}))
>>> l_dict['new_cfg'] == cfg
True
```

## 自动捕获实验日志
对深度学习任务，我们建议用`init_env`代替`load_config`，在加载配置之余，`init_env`还可以初始化深度学习环境，譬如设置 torch 设备、梯度、随机种子、分布式训练：

```python
from alchemy_cat.torch_tools import init_env

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    
    device, cfg = init_env(config_path=args.config,             # config file path，read to `cfg`
                           is_cuda=True,                        # if True，`device` is cuda，else cpu
                           is_benchmark=bool(args.benchmark),   # torch.backends.cudnn.benchmark = is_benchmark
                           is_train=True,                       # torch.set_grad_enabled(is_train)
                           experiments_root="experiment",       # root of experiment dir
                           rand_seed=True,                      # set python, numpy, torch rand seed. If True, read cfg.rand_seed as seed, else use actual parameter as rand seed. 
                           cv2_num_threads=0,                   # set cv2 num threads
                           verbosity=True,                      # print more env init info
                           log_stdout=True,                     # where fork stdout to log file
                           loguru_ini=True,                     # config a pretty loguru format
                           reproducibility=False,               # set pytorch to reproducible mode
                           local_rank=...,                      # dist.init_process_group(..., local_rank=local_rank)
                           silence_non_master_rank=True,        # if True, non-master rank will not print to stdout, but only log to file
                           is_debug=bool(args.is_debug))        # is debug mode
```
如果设置`log_stdout=True`，`init_env`还会将`sys.stdout`、`sys.stderr` fork 一份到日志文件`cfg.rslt_dir/{local-time}.log`中。这不会干扰正常的`print`，但所有屏幕输出都会同时被记录到日志。因此，不再需要手动写入日志，屏幕所见即日志所得。

更详细用法可参见`init_env`的 docstring。

## 属性字典
如果您是 [addict](https://github.com/mewwts/addict) 的用户，我们的`ADict`可以作为`addict.Dict`的 drop-in replacement：`from alchemy_cat.dl_config import ADict as Dict`。

`ADict` 实现了 `addict.Dict` 的所有接口，但重新实现了所有方法，优化了执行效率，覆盖了更多 corner case（如循环引用）。`Config`其实就是`ADict`的子类。

如果您没有使用过`addict`，可以考虑阅读这份[文档](https://github.com/mewwts/addict)。研究型代码常常会传递复杂的字典结构，`addict.Dict`或`ADict`支持属性读写字典，非常适合处理嵌套字典。

## 循环引用
`ADict`和`Config`的初始化、继承、组合需要用到一种名为`branch_copy`的操作，其介于浅拷贝和深拷贝之间，即拷贝树结构，但不拷贝叶节点。`ADict.copy`，`Config.copy`，`copy.copy(cfg)`均会调用`branch_copy`，而非`dict`的`copy`方法。

理论上`ADict.branch_copy`能够处理循环引用情况，譬如：
```text
>>> dic = {'num': 0,
           'lst': [1, 'str'],
           'sub_dic': {'sub_num': 3}}
>>> dic['lst'].append(dic['sub_dic'])
>>> dic['sub_dic']['parent'] = dic
>>> dic
{'num': 0,
 'lst': [1, 'str', {'sub_num': 3, 'parent': {...}}],
 'sub_dic': {'sub_num': 3, 'parent': {...}}}

>>> adic = ADict(dic)
>>> adic.sub_dic.parent is adic is not dic
True
>>> adic.lst[-1] is adic.sub_dic is not dic['sub_dic']
True
```
对`ADict`不同，`Config`的数据结构是双向树，而循环引用将成环。为避免成环，若子配置树被多次挂载到不同父配置，子配置树会先拷贝得到一棵独立的配置树，再进行挂载。正常使用下，配置树中不会出现循环引用。

总而言之，尽管循环引用是被支持的，不过即没有必要，也不推荐使用。

## 遍历配置树
`Config.named_branches`和`Config.named_ckl`分别遍历配置树的所有分支和叶节点（所在的分支、键名和值）：
```text
>>> list(cfg.named_branches) 
[('', {'foo': {'bar': {'a': 1}},  
       'bar': {'foo': {'b': ['str1', 'str2']}},  
       'whole': {}}),
 ('foo', {'bar': {'a': 1}}),
 ('foo.bar', {'a': 1}),
 ('bar', {'foo': {'b': ['str1', 'str2']}}),
 ('bar.foo', {'b': ['str1', 'str2']}),
 ('whole', {})]
 
>>> list(cfg.ckl)
[({'a': 1}, 'a', 1), ({'b': ['str1', 'str2']}, 'b', ['str1', 'str2'])]
```

## 惰性继承
```text
>>> from alchemy_cat.dl_config import Config
>>> cfg = Config(caps='configs/mnist/base,sched_from_addon/cfg.py')
>>> cfg.loader.ini.batch_size = 256
>>> cfg.sched.epochs = 15
>>> print(cfg)

cfg = Config()
cfg.set_whole(False).set_attribute('_cfgs_update_at_parser', ('configs/mnist/base,sched_from_addon/cfg.py',))
# ------- ↓ LEAVES ↓ ------- #
cfg.loader.ini.batch_size = 256
cfg.sched.epochs = 15
```
继承时，父配置`caps`不会被立即更新过来，而是等到`load_config`时才会被加载。惰性继承使得配置系统可以鸟瞰整条继承链，少数功能有赖于此。

## 协同Git
由于`config C + algorithm code A ——> reproducible experiment E(C, A)`，意味着当配置`C`和算法代码`A`确定时，总是能复现实验`E`。因此，建议将配置文件和算法代码一同提交到 Git 仓库中，以便日后复现实验。

我们还提供了一个[脚本](alchemy_cat/torch_tools/scripts/tag_exps.py)，运行`pyhon -m alchemy_cat.torch_tools.scripts.tag_exps -s commit_ID -a commit_ID`，将交互式地列出该 commit 新增的配置，并按照配置路径给 commit 打上标签。这有助于快速回溯历史上某个实验的配置和算法。

## 为子任务分配显卡
`Cfg2TuneRunner`的`work`函数有时需要给子进程分配显卡。`allocate_cuda_by_group_rank`可按照`pkl_idx`，分配空闲的显卡：
```python
from alchemy_cat.torch_tools import allocate_cuda_by_group_rank

# ... Code before

@runner.register_work_fn  # How to run config
def work(pkl_idx: int, cfg: Config, cfg_pkl: str, cfg_rslt_dir: str) -> ...:
    current_cudas, env_with_current_cuda = allocate_cuda_by_group_rank(group_rank=pkl_idx, group_cuda_num=2, block=True, verbosity=True)
    subprocess.run([sys.executable, 'train.py', '-c', cfg_pkl], env=env_with_current_cuda)

# ... Code after
```
`group_rank`一般为`pkl_idx`，`group_cuda_num`为子任务所需显卡数量。`block`为`True`时，若分配的显卡被占用，会阻塞直到有空闲。`verbosity`为`True`时，会打印阻塞情况。

返回值`current_cudas`是一个列表，包含了分配的显卡号。`env_with_current_cuda`是设置了`CUDA_VISIBLE_DEVICES`的环境变量字典，可直接传入`subprocess.run`的`env`参数。

## 匿名函数无法 pickle 问题
`Cfg2Tune`生成的子配置会被 pickle 保存。然而，若`Cfg2Tune`定义了形似`DEP(lambda c: ...)`的依赖项，所存储的匿名函数无法被 pickle。变通方法有：
* 配合装饰器`@Config.set_DEP`，将依赖项的计算函数定义为一个全局函数。
* 将依赖项的计算函数定义在一个独立的模块中，然后再传递给`DEP`。
* 在父配置`caps`中定义依赖项。由于继承的处理是惰性的，`Cfg2Tune`生成的子配置暂时不包含依赖项。
* 如果依赖源是可调参数，可使用特殊的依赖项`P_DEP`，它将于`Cfg2Tune`生成子配置后、保存为 pickle 前解算。

## 关于继承的更多技巧

### 继承时删除
`Config.empty_leaf()`结合了`Config.clear()`和`Config.set_whole()`，可以得到一棵空且 "whole" 的子树。这常用于在继承时表示『删除』语义，即用一个空配置，覆盖掉基配置的某颗子配置树。

### `update`方法
`cfg`是一个`Config`实例，`base_cfg`是一个`dict`实例，`cfg.dict_update(base_cfg)`、`cfg.update(base_cfg)`、`cfg |= base_cfg`的效果与让`Config(base_cfg)`继承`cfg`类似。

`cfg.dict_update(base_cfg, incremental=True)`则确保只做增量式更新——即只会增加`cfg`中不存在的键，而不会覆盖已有键。

</details>
