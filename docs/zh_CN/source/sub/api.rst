.. _api:

****************************
API: 与YLearn交互
****************************

.. list-table:: 一体化的API

    * - 类名
      - 描述
    * - :py:class:`Why`
      - 一个API封装了YLearn中几乎所有的东西，比如 *识别因果效应* 和 *给一个训练过的估计模型打分* 。 它给用户提供了简单和有效的方法使用YLearn。

.. list-table:: 因果结构发现

    * - 类名
      - 描述
    * - :py:class:`CausalDiscovery`
      - 发现观测数据中的因果结构。

.. list-table:: 因果模型

    * - 类名
      - 描述
    * - :py:class:`CausalGraph`
      - 表示因果结构和支持因果图其他相关的操作，例如加减图中的边。
    * - :py:class:`CausalModel`
      - 编码由 :py:class:`CausalGraph` 表示的因果关系。主要支持因果效应识别，比如后门调整。
    * - :py:class:`Prob`
      - 表示概率分布。

.. list-table:: 估计模型

    * - 类名
      - 描述
    * - :py:class:`ApproxBound`
      - 一个用于估计因果效应上下界的模型。该模型不需要无混杂条件。
    * - :py:class:`CausalTree`
      - 一个通过决策树估计因果效应的类。需要无混杂条件。
    * - :py:class:`DeepIV`
      - 具有深度神经网络的工具变量。必须提供工具变量的名字。
    * - :py:class:`NP2SLS`
      - 无参数的工具变量。必须提供工具变量的名字。
    * - :py:class:`DoubleML`
      - 双机器学习模型用于估计CATE。需要无混杂条件。
    * - :py:class:`DoublyRobust` and :py:class:`PermutedDoublyRobust`
      - 双鲁棒方法用于估计CATE。置换的版本考虑了所有可能的治疗控制对。需要无混杂条件且治疗必须是离散的。
    * - :py:class:`SLearner` and :py:class:`PermutedSLearner`
      - SLearner。 置换的版本考虑了所有可能的治疗控制对。需要无混杂条件且治疗必须是离散的。
    * - :py:class:`TLearner` and :py:class:`PermutedTLearner`
      - 使用了多个机器学习模型的TLearner。置换的版本考虑了所有可能的治疗控制对。需要无混杂条件且治疗必须是离散的。
    * - :py:class:`XLearner` and :py:class:`PermutedXLearner`
      - 使用了多个机器学习模型的XLearner。置换的版本考虑了所有可能的治疗控制对。需要无混杂条件且治疗必须是离散的。
    * - :py:class:`RLoss`
      - 通过测量估计模型的效果得到效果分。需要无混杂条件。

.. list-table:: 策略

    * - 类名
      - 描述
    * - :py:class:`PolicyTree`
      - 一个通过树模型和最大化因果效应来找到最优的策略的类。

.. list-table:: 解释器

    * - 类名
      - 描述
    * - :py:class:`CEInterpreter`
      - 一个使用决策树模型的对象，用于解释估计的CATE。
    * - :py:class:`PolicyInterpreter`
      - 一个用于解释由一些 :py:class:`PolicyModel` 给出的策略的对象。

