*****************************
Forest Estimator Models
*****************************


Random forest is a widely used algorithm in machine learning. Many empirical properties of random forest including stability and the ability of flexible adaptation
to complicated forms have made random forest and its variants as popular and reliable choices in a lot of tasks. It is then a natural and crucial idea to extend tree 
based models for causal effect estimation such as causal tree to forest based ones. These works are pioneered by [Athey2018]_. Similar to the case of machine
learning, typically for causal effect estimation, forest estimator models have better performance than tree models while sharing equivalent interpretability and other
advantages. Thus it is always recommended to try these estimator models first.

In YLearn, we currently cover three types of forest estimator models for causal effect estimation under the unconfoundness asssumption:
    
    .. toctree::
        :maxdepth: 2
        
        forest_based/grf.rst
        forest_based/cf.rst
        forest_based/ctcf.rst
