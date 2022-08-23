# TODO: In the current implementation, we believe that the tree model can be a
# valid model for predicting the causal effects estimated by the estimator model. Therefore,
# we can directly apply the fitted tree model to new test dataset and add support
# for such method. In the later version, we may need to verify the correctness of
# doing so and modify the method accordingly.
import re

import numpy as np
import pandas as pd

from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._export import _MPLTreeExporter
from sklearn.tree._reingold_tilford import buchheim


from ylearn.estimator_model.utils import convert2array


class _CateTreeExporter(_MPLTreeExporter):

    def __init__(self, node_dict, include_uncertainty=False, uncertainty_level=0.1,
                 *args, treatment_names=None, **kwargs):
        self.node_dict = node_dict
        self.include_uncertainty = include_uncertainty
        self.uncertainty_level = uncertainty_level
        self.treatment_names = treatment_names
        super().__init__(*args, **kwargs)

    def get_color(self, value):
        # Find the appropriate color & intensity for a node
        if self.colors["bounds"] is None:
            # Classification tree
            color = list(self.colors["rgb"][np.argmax(value)])
            sorted_values = sorted(value, reverse=True)
            if len(sorted_values) == 1:
                alpha = 0
            else:
                alpha = (sorted_values[0] - sorted_values[1]) / (1 - sorted_values[1])
        else:
            # Regression tree or multi-output
            color = list(self.colors["rgb"][0])
            alpha = (value - self.colors["bounds"][0]) / (
                self.colors["bounds"][1] - self.colors["bounds"][0]
            )
        # unpack numpy scalars
        alpha = float(alpha)
        # compute the color as alpha against white
        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        # Return html color code in #RRGGBB format
        return "#%02x%02x%02x" % tuple(color)

    def get_fill_color(self, tree, node_id):

        # Fetch appropriate color for node
        if 'rgb' not in self.colors:
            # red for negative, green for positive
            self.colors['rgb'] = [(233, 150, 60), (6, 42, 220)]

        # in multi-target use mean of targets
        tree_min = np.min(np.mean(tree.value, axis=1)) - 1e-12
        tree_max = np.max(np.mean(tree.value, axis=1)) + 1e-12

        node_val = np.mean(tree.value[node_id])

        if node_val > 0:
            value = [max(0, tree_min) / tree_max, node_val / tree_max]
        elif node_val < 0:
            value = [node_val / tree_min, min(0, tree_max) / tree_min]
        else:
            value = [0, 0]

        return self.get_color(value)

    def export(self, decision_tree, ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.text import Annotation

        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_axis_off()
        my_tree = self._make_tree(0, decision_tree.tree_, decision_tree.criterion)
        draw_tree = buchheim(my_tree)

        # important to make sure we're still
        # inside the axis after drawing the box
        # this makes sense because the width of a box
        # is about the same as the distance between boxes
        max_x, max_y = draw_tree.max_extents() + 1
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height

        scale_x = ax_width / max_x
        scale_y = ax_height / max_y
        self.recurse(draw_tree, decision_tree.tree_, ax, max_x, max_y, text_pos=0)

        anns = [ann for ann in ax.get_children() if isinstance(ann, Annotation)]

        # update sizes of all bboxes
        renderer = ax.figure.canvas.get_renderer()

        for ann in anns:
            ann.update_bbox_position_size(renderer)

        if self.fontsize is None:
            # get figure to data transform
            # adjust fontsize to avoid overlap
            # get max box width and height
            extents = [ann.get_bbox_patch().get_window_extent() for ann in anns]
            max_width = max([extent.width for extent in extents])
            max_height = max([extent.height for extent in extents])
            # width should be around scale_x in axis coordinates
            size = anns[0].get_fontsize() * min(
                scale_x / max_width, scale_y / max_height
            )
            for ann in anns:
                ann.set_fontsize(size)

        return anns

    def recurse(self, node, tree, ax, max_x, max_y, text_pos,  depth=0):
        import matplotlib.pyplot as plt

        kwargs = dict(
            bbox=self.bbox_args.copy(),
            ha="center",
            va="center",
            zorder=100 - 10 * depth,
            xycoords="axes fraction",
            arrowprops=self.arrow_args.copy(),
        )
        kwargs["arrowprops"]["edgecolor"] = plt.rcParams["text.color"]

        if self.fontsize is not None:
            kwargs["fontsize"] = self.fontsize

        # offset things by .5 to center them in plot
        xy = ((node.x + 0.5) / max_x, (max_y - node.y - 0.5) / max_y)

        if self.max_depth is None or depth <= self.max_depth:
            if self.filled:
                kwargs["bbox"]["fc"] = self.get_fill_color(tree, node.tree.node_id)
            else:
                kwargs["bbox"]["fc"] = ax.get_facecolor()

            if node.parent is None:
                # root
                ax.annotate(node.tree.label, xy, **kwargs)
            else:
                xy_parent = (
                    (node.parent.x + 0.5) / max_x,
                    (max_y - node.parent.y - 0.5) / max_y,
                )
                ax.annotate(node.tree.label, xy_parent, xy, **kwargs)

                text_pos_mapping = {
                    1: ('yes', 'right', -0.015),
                    -1: ('no', 'left', 0.015)
                    # 0: ('center', 0, '')
                }

                if text_pos in [1, -1]:
                    text_pos_config = text_pos_mapping[text_pos]
                    ax.text((xy_parent[0] - xy[0])/2 + xy[0] + text_pos_config[2],
                            (xy_parent[1] - xy[1])/2 + xy[1],
                            text_pos_config[0], va="center", ha=text_pos_config[1], rotation=0)

            n_children = len(node.children)
            for i, child in enumerate(node.children):
                if i == 0:
                    next_text_pos = 1
                elif i == n_children - 1:
                    next_text_pos = -1
                else:
                    next_text_pos = 0
                self.recurse(child, tree, ax, max_x, max_y, text_pos=next_text_pos, depth=depth + 1)

        else:
            xy_parent = (
                (node.parent.x + 0.5) / max_x,
                (max_y - node.parent.y - 0.5) / max_y,
            )
            kwargs["bbox"]["fc"] = "grey"
            ax.annotate("\n  (...)  \n", xy_parent, xy, **kwargs)

    def node_replacement_text(self, tree, node_id, criterion):
        # Write node mean CATE
        node_info = self.node_dict[node_id]
        node_string = 'CATE mean' + self.characters[4]
        value_text = ""
        mean = node_info['mean']
        if hasattr(mean, 'shape') and (len(mean.shape) > 0):
            if len(mean.shape) == 1:
                for i in range(mean.shape[0]):
                    value_text += "{}".format(np.around(mean[i], self.precision))
                    if 'ci' in node_info:
                        value_text += " ({}, {})".format(np.around(node_info['ci'][0][i], self.precision),
                                                         np.around(node_info['ci'][1][i], self.precision))
                    if i != mean.shape[0] - 1:
                        value_text += ", "
                value_text += self.characters[4]
            elif len(mean.shape) == 2:
                for i in range(mean.shape[0]):
                    for j in range(mean.shape[1]):
                        value_text += "{}".format(np.around(mean[i, j], self.precision))
                        if 'ci' in node_info:
                            value_text += " ({}, {})".format(np.around(node_info['ci'][0][i, j], self.precision),
                                                             np.around(node_info['ci'][1][i, j], self.precision))
                        if j != mean.shape[1] - 1:
                            value_text += ", "
                    value_text += self.characters[4]
            else:
                raise ValueError("can only handle up to 2d values")
        else:
            value_text += "{}".format(np.around(mean, self.precision))
            if 'ci' in node_info:
                value_text += " ({}, {})".format(np.around(node_info['ci'][0], self.precision),
                                                 np.around(node_info['ci'][1], self.precision))
            value_text += self.characters[4]
        node_string += value_text

        # Write node std of CATE
        node_string += "CATE std" + self.characters[4]
        std = node_info['std']
        value_text = ""
        if hasattr(std, 'shape') and (len(std.shape) > 0):
            if len(std.shape) == 1:
                for i in range(std.shape[0]):
                    value_text += "{}".format(np.around(std[i], self.precision))
                    if i != std.shape[0] - 1:
                        value_text += ", "
            elif len(std.shape) == 2:
                for i in range(std.shape[0]):
                    for j in range(std.shape[1]):
                        value_text += "{}".format(np.around(std[i, j], self.precision))
                        if j != std.shape[1] - 1:
                            value_text += ", "
                    if i != std.shape[0] - 1:
                        value_text += self.characters[4]
            else:
                raise ValueError("can only handle up to 2d values")
        else:
            value_text += "{}".format(np.around(std, self.precision))
        node_string += value_text
        return node_string

    def node_to_str(self, tree, node_id, criterion):
        text = super().node_to_str(tree, node_id, criterion)
        replacement = self.node_replacement_text(tree, node_id, criterion)
        if replacement is not None:
            # HACK: it's not optimal to use a regex like this, but the base class's node_to_str doesn't expose any
            #       clean way of achieving this
            text = re.sub("value = .*(?=" + re.escape(self.characters[5]) + ")",
                          # make sure we don't accidentally escape anything in the substitution
                          replacement.replace('\\', '\\\\'),
                          text,
                          flags=re.S)
        return text


class CEInterpreter:
    """
    Many parameters are similar to those of BaseDecisionTree of sklearn.

    Parameters
    ----------
    criterion : {"squared_error", "friedman_mse", "absolute_error", \
            "poisson"}, default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.        

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
        `ceil(min_samples_split * n_samples)` are the minimum
        number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
            
            - If int, then consider `min_samples_leaf` as the minimum number.
            - If float, then `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)` are the minimum number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
        `int(max_features * n_features)` features are considered at each
        split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

    random_state : int
        Controls the randomness of the estimator.

    max_leaf_nodes : int, default to None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    ccp_alpha : non-negative float, default to 0.0
        Value for pruning the tree. #TODO: not implemented yet.
    """
    def __init__(
        self, *,
        criterion='squared_error',
        splitter='best',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        ccp_alpha=0.0,
    ):
        self._is_fitted = False
        self.treatment = None
        self.outcome = None

        self._tree_model = DecisionTreeRegressor(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha
        )

        self._est_model = None
        self.node_dict_ = None

    def fit(
        self,
        data,
        est_model,
        **kwargs
    ):
        #TODO: make this more compatible with that in the policy_interpreter
        """Fit the CEInterpreter model to interpret the causal effect estimated
        by the est_model on data.

        Parameters
        ----------
        data : pandas.DataFrame
            The input samples for the est_model to estimate the causal effects
            and for the CEInterpreter to fit.

        est_model : estimator_model
            est_model should be any valid estimator model of ylearn which was 
            already fitted and can estimate the CATE.
        """
        assert est_model._is_fitted

        self._est_model = est_model

        self.covariate = est_model.covariate
        assert self.covariate is not None, 'Need covariate to interpret the causal effect.'

        v = convert2array(data, self.covariate)[0]
        n = v.shape[0]

        v = self._transform_data(v)

        self._v = v

        causal_effect = est_model.estimate(data=data, quantity=None, **kwargs)

        self._tree_model.fit(v, causal_effect.reshape((n, -1)))

        paths = self._tree_model.decision_path(v)

        node_dict = {}
        for node_id in range(paths.shape[1]):
            mask = paths.getcol(node_id).toarray().flatten().astype(bool)
            # Xsub = v[mask]
            cate_node = causal_effect[mask]
            node_dict[node_id] = {'mean': np.mean(cate_node, axis=0), 'std': np.std(cate_node, axis=0)}

        self.node_dict_ = node_dict
        self._is_fitted = True

        return self

    def interpret(self, *, v=None, data=None):
        """Interpret the fitted model in the test data.

        Parameters
        ----------
        v : numpy.ndarray, optional
            The test covariates, by default None
        data : DataFrame, optional
            The test data in the form of the DataFrame. The model will only use this if v is set as None. In this case, if data is also None, then the data used for trainig will be used, by default None

        Returns
        -------
        dict
            The interpreted results for all examples.
        """
        assert self._is_fitted, 'The model is not fitted yet. Please use the fit method first.'

        v = self._check_features(v=v, data=data)

        node_indicator = self._tree_model.decision_path(v)
        leaf_id = self._tree_model.apply(v)
        feature = self._tree_model.tree_.feature
        threshold = self._tree_model.tree_.threshold

        result_dict = {}
        
        for sample_id, sample in enumerate(v):
            result_dict[f'sample_{sample_id}'] = ''
            node_index = node_indicator.indices[
                node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
            ]

            for node_id in node_index:
                if leaf_id[sample_id] == node_id:
                    continue

                if v[sample_id, feature[node_id]] <= threshold[node_id]:
                    thre_sign = '<='
                else:
                    thre_sign = '>'

                feature_id = feature[node_id]
                value = v[sample_id, feature_id]
                thre = threshold[node_id]
                result_dict[f'sample_{sample_id}'] += f'decision node {node_id}: (covariate [{sample_id}, {feature_id}] = {value})' \
                    f' {thre_sign} {thre} \n'
        
        return result_dict

    def _check_features(self, *, v=None, data=None):
        """Validate the training data on predict (probabilities)."""

        if v is not None:
            v = v.reshape(-1, 1) if v.ndim == 1 else v
            assert v.shape[1] == self._tree_model.n_features_in_
            v = v.astype(np.float32)

            return v

        if data is None:
            v = self._v
        else:
            assert isinstance(data, pd.DataFrame)

            v = convert2array(data, self.covariate)[0]
            if hasattr(self, 'cov_transformer'):
                v = self.cov_transformer.transform(v)

            assert v.shape[1] == self._tree_model.n_features_in_
            v = v.reshape(-1, 1) if v.ndim == 1 else v

        v = v.astype(np.float32)

        return v

    def _transform_data(self, data):
        if hasattr(self._est_model, 'covariate_transformer'):
            ct = self._est_model.covariate_transformer
            if ct is not None:
                return ct.transform(data)
            else:
                return data
        else:
            return data

    def decide(self, data):
        data_t = self._transform_data(data)
        return self._tree_model.predict(data_t)

    def plot(
        self, *,
        feature_names=None,
        max_depth=None,
        class_names=None,
        label='all',
        filled=True,
        node_ids=False,
        proportion=False,
        rounded=False,
        precision=3,
        ax=None,
        fontsize=None
    ):
        """Plot a policy tree.
        The sample counts that are shown are weighted with any sample_weights that
        might be present.
        The visualization is fit automatically to the size of the axis.
        Use the ``figsize`` or ``dpi`` arguments of ``plt.figure``  to control
        the size of the rendering.

        Parameters
        ----------        
        max_depth : int, default=None
            The maximum depth of the representation. If None, the tree is fully
            generated.
        
        class_names : list of str or bool, default=None
            Names of each of the target classes in ascending numerical order.
            Only relevant for classification and not supported for multi-output.
            If ``True``, shows a symbolic representation of the class name.
        
        label : {'all', 'root', 'none'}, default='all'
            Whether to show informative labels for impurity, etc.
            Options include 'all' to show at every node, 'root' to show only at
            the top root node, or 'none' to not show at any node.
        
        filled : bool, default=False
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.
        
        impurity : bool, default=True
            When set to ``True``, show the impurity at each node.
        
        node_ids : bool, default=False
            When set to ``True``, show the ID number on each node.
        
        proportion : bool, default=False
            When set to ``True``, change the display of 'values' and/or 'samples'
            to be proportions and percentages respectively.
        
        rounded : bool, default=False
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.
        
        precision : int, default=3
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.
        
        ax : matplotlib axis, default=None
            Axes to plot to. If None, use current axis. Any previous content
            is cleared.
        
        fontsize : int, default=None
            Size of text font. If None, determined automatically to fit figure.
        
        Returns
        -------
        annotations : list of artists
            List containing the artists for the annotation boxes making up the
            tree.
        """
        assert self._is_fitted

        impurity = False

        if feature_names == None:
            feature_names = self.covariate

        exporter = _CateTreeExporter(
            self.node_dict_,
            max_depth=max_depth,
            feature_names=feature_names,
            class_names=class_names,
            label=label,
            filled=filled,
            impurity=impurity,
            node_ids=node_ids,
            proportion=proportion,
            rounded=rounded,
            precision=precision,
            fontsize=fontsize,
        )
        return exporter.export(self._tree_model, ax=ax)
