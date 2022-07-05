***************************************
Why: 一个一体化的因果学习API
***************************************

想要轻松使用YLearn？尝试一体化的API `Why`!

`Why` 是一个封装了YLearn中几乎所有东西的API，比如 *识别因果效应* 和 *给一个训练好的估计器模型打分*。它提供给用户一个简单且有效的方式来使用
我们的包：能够直接传入你仅有的东西，数据到 `Why` 中并使用它的多个方法，而不是学习多个概念，比如在能够找到藏在你的数据中有趣的信息之前学习调整集合。
`Why` 被设计成能够执行因果推断的完整流程：给出数据，它首先尝试发现因果图，如果没有提供的话。接着它尝试找到可能作为治疗的变量和识别因果效应。在此之后，一个
合适的估计器模型将被训练用以估计因果效应。最终，估计对于每一个个体最佳的选项的策略。

.. figure:: flow.png

    `Why` 能够帮助因果推断的完整流程的几乎每一个部分。

类结构
================

.. class:: ylearn._why.Why(discrete_outcome=None, discrete_treatment=None, identifier='auto', discovery_model=None, discovery_options=None, estimator='auto', estimator_options=None, random_state=None)

    一个一体化的因果学习API。

    :param bool, default=infer from outcome discrete_outcome:
    :param bool, default=infer from the first treatment discrete_treatment:
    :param str, default=auto' identifier: 可用的选项： 'auto' 或 'discovery'
    :param str, optional, default=None discovery_model:
    :param dict, optional, default=None discovery_options: 参数（键值对）来初始化发现模型
    :param str, optional, default='auto' estimator: 一个合理的EstimatorModel的名字。 也可以传入一个合理的估计器模型的实例。
    :param dict, optional, default=None estimator_options: 参数（键值对）来初始化估计器模型
    :param int, optional, default=None random_state:
    
    .. py:attribute:: `feature_names_in_`
        
        在 `fit` 时看到的特征的名字的列表
    
    .. py:attribute:: outcome_

        结果的名字

    .. py:attribute:: treatment_

        在 `fit` 时识别的治疗的名字的列表
    
    .. py:attribute:: adjustment_

        在 `fit` 时识别的调整的名字的列表
    
    .. py:attribute:: covariate_

        在 `fit` 时识别的协变量的名字的列表
    
    .. py:attribute:: instrument_

        在 `fit` 时识别的工具的名字的列表
    
    .. py:attribute:: identifier_

        `identifier` 对象或者None. 用于识别治疗/调整/协变量/工具，如果在 `fit` 时没有被指明

    .. py:attribute:: y_encoder_

        `LabelEncoder` 对象或者None. 用于编码结果，如果它的dtype不是数字的。
    
    .. py:attribute:: preprocessor_
        
        `Pipeline` 对象在 `fit` 时预处理数据

    .. py:attribute:: estimators_

        对于每个治疗的估计器字典，其中键是治疗的名字，值是 `EstimatorModel` 对象

    .. py:method:: fit(data, outcome, *, treatment=None, adjustment=None, covariate=None, instrument=None, treatment_count_limit=None, copy=True, **kwargs)

        拟合Why对象，步骤：
            
            1. 编码结果如果它的dtype不是数字的
            2. 识别治疗和调整/协变量/工具
            3. 预处理数据
            4. 拟合因果估计器
        
        :returns: 拟合的 :py:class:`Why` 。
        :rtype: :py:class:`Why` 的实例

    .. py:method:: identify(data, outcome, *, treatment=None, adjustment=None, covariate=None, instrument=None, treatment_count_limit=None)

        识别治疗和调整/协变量/工具。

        :returns: 识别的治疗，调整，协变量，工具
        :rtypes: tuple

    .. py:method:: causal_graph()

        获得识别的因果图。

        :returns: 识别的因果图
        :rtype: :py:class:`CausalGraph` 的实例

    .. py:method:: causal_effect(test_data=None, treat=None, control=None)

        估计因果效应。

        :returns: 所有治疗的因果效应
        :rtype: pandas.DataFrame
    
    .. py:method:: individual_causal_effect(test_data, treat=None, control=None)

        为每一个个体估计因果效应。

        :returns: 对于每一个治疗，个体的因果效应
        :rtype: pandas.DataFrame
    
    .. py:method:: whatif(data, new_value, treatment=None)

        获得反事实预测当治疗从它的对应的观测变为new_value。

        :returns: 反事实预测
        :rtype: pandas.Series
 
    .. py:method:: score(test_data=None, treat=None, control=None, scorer='auto')

        :returns: 估计器模型的分数
        :rtype: float
   
    .. py:method:: policy_tree(data, control=None, **kwargs)

        获得策略树

        :returns: 拟合的 :py:class:`PolicyTree` 的实例。
        :rtype: :py:class:`PolicyTree` 的实例

    .. py:method:: policy_interpreter(data, control=None, **kwargsa)

        获得策略解释器

        :returns: 拟合的 :py:class:`PolicyInterpreter` 的实例。
        :rtype: :py:class:`PolicyInterpreter` 的实例

    .. py:method:: plot_causal_graph()

        绘制因果图。
    
    .. py:method:: plot_policy_tree(Xtest, control=None, **kwargs)

        绘制策略树。
    
    .. py:method:: plot_policy_interpreter(data, control=None, **kwargs)

        绘制解释器。