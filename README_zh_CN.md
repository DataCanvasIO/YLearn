<h1 align="center">
<img src="./fig/YLearn1.png" width="400" align=center/>
</h1><br>

[English](https://github.com/DataCanvasIO/YLearn/blob/main/README.md)

**YLearn**是一个因果学习算法工具包，这个名字是“learn why”的谐音。它主要支持因果学习任务中的各类相关任务，从因果效应识别（causal effect idenfitication），到因果效应估计（causal effect estimation），到因果发现（causal discovery）等等，都可以通过YLearn实现，最大程度地降低了学习一个因果工具的成本。

**Documentation website**: <https://ylearn.readthedocs.io>

**中文文档地址**：<https://ylearn.readthedocs.io/zh_CN/latest/>

## 安装

### Pip

最直接的安装方式可以通过`pip`进行安装:

```bash
pip install ylearn
```

注意：YLearn在绘制因果图时需要用到`Graphviz`，所以请在运行YLearn之前安装它。关于`Graphviz`的安装方式请参考 https://graphviz.org/download/ 。

###  Docker

如果您拥有Docker环境的话，可下载我们发布的Docker镜像。YLearn的Docker镜像中包括如下内容：

* Python 3.8
* YLearn 及其依赖包
* JupyterLab

下载镜像:

```bash
docker pull datacanvas/ylearn
```

运行镜像:

```bash
docker run -ti -e NotebookToken="your-token" -p 8888:8888 datacanvas/ylearn
```
然后通过浏览器访问 “ http://&lt;ip-addr&gt;:8888 ”，输入您设置的token就可以开始使用YLearn了。


## YLearn概览

机器学习在近些年来有了很大的发展，也在很多场景中取得了成功，但是这些场景多是用作预测相关等任务，例如猫狗图片识别这一经典案例。但是，有很多自然产生并且常见的场景任务是机器学习无法胜任的，这其中一个直观的例子是策略评估场景下所谓的 **“反事实问题”**（counterfactual question）：如果当初采用的是另外的策略，现在面临的结果又会是怎样的呢？众所周知，这些反事实造成的结果是永远无法被观测到的，所以在这一类场景下，作为预测工具的机器学习模型便失效了。处理这些机器学习模型无法直接胜任的事情一定程度上也促进了最近以来对因果推断相关知识的需求，更为重要的方面则是因果推断与机器学习的交叉，YLearn正是在这种需求下开发的。

因果推断直接在干预（interventions）下对结果进行建模，并且形成了反事实推断等能力。在今天大发展的机器学习模型帮助下，因果推断现在可以更方便地使用多种多样的方式从观测数据中去提取 **因果相关的结论**，而不是像过去一样更依赖于精心设计的各类随机实验，因为一些成本等方面的原因，这些实验有时甚至无法开展。

一个典型的完整因果推断流程主要由三个部分组成。*第一*，数据中的因果结构应当首先被学习和发现，用作这一任务的手段通常被称为因果发现（causal discovery）。这些被发现的因果关系会被表示为因果结构公式（structural causal models, SCM）或因果图（一种有向无环图，directed acyclic graphs, DAG）。*第二*，接下来我们需要将我们感兴趣的因果问题中的量用因果变量（causal estimand）表示，其中一个例子是平均治疗效应（average treatment effect, ATE）。这些因果变量接下来会通过因果效应识别转化为统计变量（statistical estimand），这是因为因果变量无法从数据中直接估计，只有识别后的因果变量才可以从数据中被估计出来。*最后*，我们需要选择合适的因果估计模型从数据中去学些这些被识别后的因果变量。完成这些事情之后，诸如策略估计问题和反事实问题等因果问题也可以被解决了。

YLearn 实现了最近文献中发展出的多个因果推断相关算法并且致力于**在机器学习的帮助下支持因果推断中从因果发现到因果效应估计**等各方各面的相关内容，尤其是当有很多观测得到的数据时，这一目的会更有前景。

### YLearn概念

![Concepts in YLearn](./fig/structure_ylearn.png#pic_center)

YLearn 有5个主要的因果推断相关概念，如下所示

1. *CausalDiscovery*. 主要作用是发现数据中的因果结构

2. *CausalModel*. 将因果结构和因果关系用``CausalGraph``表示，并且依靠``CausalModel``实现各类因果操作（如因果效应识别）

3. *EstimatorModel*. 用多种估计方式从数据中估计因果效应

4. *Policy*. 为每个个体选取最合适的策略方案

5. *Interpreter*. 解释所估计得到的因果效应和策略方案

这些不同的部分通过组合，可以完成一个完整的因果学习相关流程，为了方便使用，YLearn也将它们一起封装在`Why`这个统一的API借口中。

### YLearn流程

![A typical pipeline in YLearn](./fig/flow.png#pic_center)
*YLearn中的因果推断流程*

上图展示了一个YLearn中的完整因果推断流程，我们介绍如下。从用户给定的训练数据开始，我们首先:

1. 使用 `CausalDiscovery` 去发现数据中的因果关系和因果结构，它们会以 `CausalGraph` 的形式表示和存在。
2. 这些因果图接下来会被输入进 `CausalModel`, 在这里用户感兴趣的因果变量会通过因果效应识别转化为相应的可被估计的统计变量（也叫识别后的因果变量）。
3. 一个特定的 `EstimatorModel` 此时会在训练集中训练，得到训练好的估计模型，用来从数据中估计识别后的因果变量。
4. 这个（些）训练好的 `EstimatorModel` 就可以被用来在测试数据集上估计各类不同的因果效应，同时也可以被用来作因果效应解释或策略方案的制定。

下面的流程图可以方便地帮助用户决定自己的使用步骤

![Helpful flow chart when using YLearn](./fig/flow_chart_cn.png#pic_center)

## 快速开始

这一部分中，我们将展示数个简单的YLearn使用例子，它们基本覆盖了最普遍的YLearn功能。

### 使用示例

这一部分我们展示简单的使用示例，请参阅它们对应的文档查看更多细节。

1. **表示因果图**

   在YLearn中，给定一个变量集合，与之相关的因果关系*需要一个 python `dict` 去表示变量中的因果关系*，在这个 `dict` 中，每一个 *key* 都是它相应 *value* （通常是一个`list`）中的每一个元素的 children。我们举一个最简单的例子，给定因果结构 `X <- W -> Y` ，我们首先定一个一个 python `dict` 表示相关因果结构，这个 `dict` 会被当作参数传入 `CausalGraph` 中：

    ```python
        causation = {'X': ['W'], 'W':[], 'Y':['W']}
        cg = CausalGraph(causation=causation)
    ```

    `cg` 就是我们的表示了因果关系 `X <- W -> Y` 的因果图。同时需要注意的是，如果存在 *不可观测的混淆因子*（unobserved confounder），那么除了前述的 `dict` 外，我们需要一个额外的 python `list` 去记录这些不可观测的因果结构，比如下面的因果图存在不可观测的混淆因子（绿色节点）

    <img src="./fig/graph_expun.png" width="400">


   它会首先被转化为一个有潜在混淆曲线（latent confounding arcs，下图中有两个箭头的黑色曲线）的因果图

    <img src="./fig/graph_un_arc.png" width="500">

   接着为了表示这张图，我们需要

   **(1)** 定义一个 python `dict` 表示图中可观测的部分

   **(2)** 定义一个 `list` 记录不可观测的潜在混淆曲线，其中 `list` 中的每一个元素包括一条不可观测潜在混淆曲线的两个端点:

   ```python
        from ylearn.causal_model.graph import CausalGraph
        causation_unob = {
            'X': ['Z2'],
            'Z1': ['X', 'Z2'],
            'Y': ['Z1', 'Z3'],
            'Z3': ['Z2'],
            'Z2': [], 
        }
        arcs = [('X', 'Z2'), ('X', 'Z3'), ('X', 'Y'), ('Z2', 'Y')]

        cg_unob = CausalGraph(causation=causation_unob, latent_confounding_arcs=arcs)
   ```

2. **因果效应识别**

   因果效应识别对于因果效应（包括因果变量）估计是至关重要的，这一过程可以通过 YLearn 很轻松地实现。例如，假设我们希望识别上面的因果图 `cg` 中的因果变量 `P(Y|do(X=x))`，那么我们只需要定义一个 `CausalModel` 然后调用它的 `identify()` 方法即可：

   ```python
        cm = CausalModel(causal_graph=cg)
        cm.identify(treatment={'X'}, outcome={'Y'}, identify_method=('backdoor', 'simple'))
   ```

    在上面的例子中我们使用了 *后门调整*，YLearn 也支持包括前门调整，工具变量识别，一般因果效应识别[1]（如果任意因果量可以被识别，返回识别后的结果，如果不可识别，则返回不可识别）等各类识别算法。

3. **工具变量**

   工具变量是一种因果推断中很重要的手段，利用 YLearn 去寻找工具变量十分方便直接，例如，我们有如下的因果图

    <img src="./fig/iv2.png" width="400">,

   那么我们可以按我们使用 `CausalModel` 的常用步骤来识别工具变量：（1）定义 `dict` 和 `list` 去表示因果关系；（2）定义 `CausalGraph` 的 instance 建立 YLearn 中的因果图；（3）以上一步定义的 `CausalGraph` 作为参数，定义 `CausalModel` 的 instance；（4）调用 `CausalModel` 的 `get_iv()` 寻找工具变量：

   ```python
        causation = {
            'p': [],
            't': ['p', 'l'],
            'l': [],
            'g': ['t', 'l']
        }
        arc = [('t', 'g')]
        cg = CausalGraph(causation=causation, latent_confounding_arcs=arc)
        cm = CausalModel(causal_graph=cg)
        cm.get_iv('t', 'g')
   ```

4. **因果效应估计**

   使用 YLearn 进行因果效应估计十分方便直接（与通常的机器学习模型使用方式十分类似，因为 YLearn 主要着眼于机器学习与因果推断的交叉），它是一个包括3个步骤的流程：

    * 给定 `pandas.DataFrame` 形式的数据，确定 `treatment, outcome, adjustment, covariate` 的变量名。
    * 调用 `EstimatorModel` 的 `fit()` 方法训练模型。
    * 调用 `EstimatorModel` 的 `estimate()` 方法得到估计好的因果效应

   用户可以查看文档中的[相关页面](https://ylearn.readthedocs.io/en/latest/sub/est.html#)查阅各类估计模型的细节。

5. **使用统一接口API: Why**

   为了能*以一种统一且方便的方式使用 YLearn*，YLearn 提供了一个接口 `Why`，它几乎封装了 YLearn 中的所有内容，包括因果效应识别和评估训练得到的估计模型等。在使用`Why` 的过程中，用户可以先创建一个 `Why` 的实例，然后调用 `Why` 的 `fit()` 方法训练这个实例，之后其他的各类方法（如`causal_effect()`, `score()`, `whatif()`）就可以使用了。下面的代码是一个简单的使用样例：

    ```python
        from sklearn.datasets import fetch_california_housing

        from ylearn import Why

        housing = fetch_california_housing(as_frame=True)
        data = housing.frame
        outcome = housing.target_names[0]
        data[outcome] = housing.target

        why = Why()
        why.fit(data, outcome, treatment=['AveBedrms', 'AveRooms'])

        print(why.causal_effect())
    ```

### 案例

在 notebook [CaseStudy](https://github.com/DataCanvasIO/YLearn/blob/main/example_usages/case_study_bank.ipynb) 中, 我们用一个银行客户相关的数据集进一步阐述了 `Why` 的各类用法和 YLearn 的实际场景使用举例。 请参考 [CaseStudy](https://github.com/DataCanvasIO/YLearn/blob/main/example_usages/case_study_bank.ipynb) 查看更多的细节。

## 欢迎贡献

欢迎来自社区的开发者贡献！在开始前，请先阅读 [code of conduct](CODE_OF_CONDUCT.md) 和 [contributing guidelines](CONTRIBUTING.md).

## 参考文献

[1] J. Pearl. Causality: models, reasoing, and inference.

[2] S. Shpister and J. Identification of Joint Interventional Distributions in Recursive Semi-Markovian Causal Models. *AAAI 2006*.

[3] B. Neal. Introduction to Causal Inference.

[4] M. Funk, et al. Doubly Robust Estimation of Causal Effects. *Am J Epidemiol. 2011 Apr 1;173(7):761-7.*

[5] V. Chernozhukov, et al. Double Machine Learning for Treatment and Causal Parameters. *arXiv:1608.00060.*

[6] S. Athey and G. Imbens. Recursive Partitioning for Heterogeneous Causal Effects. *arXiv: 1504.01132.*

[7] A. Schuler, et al. A comparison of methods for model selection when estimating individual treatment effects. *arXiv:1804.05146.*

[8] X. Nie, et al. Quasi-Oracle estimation of heterogeneous treatment effects. *arXiv: 1712.04912.*

[9] J. Hartford, et al. Deep IV: A Flexible Approach for Counterfactual Prediction. *ICML 2017.*

[10] W. Newey and J. Powell. Instrumental Variable Estimation of Nonparametric Models. *Econometrica 71, no. 5 (2003): 1565–78.*

[11] S. Kunzel2019, et al. Meta-Learners for Estimating Heterogeneous Treatment Effects using Machine Learning. *arXiv: 1706.03461.*

[12] J. Angrist, et al. Identification of causal effects using instrumental variables. *Journal of the American Statistical Association*.

[13] S. Athey and S. Wager. Policy Learning with Observational Data. *arXiv: 1702.02896.*

[14] P. Spirtes, et al. Causation, Prediction, and Search.

[15] X. Zheng, et al. DAGs with NO TEARS: Continuous Optimization for Structure Learning. *arXiv: 1803.01422.*
