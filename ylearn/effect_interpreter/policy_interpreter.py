from ..policy.policy_model import PolicyTree

import re

from ..utils._common import convert2array
from sklearn.tree._export import _BaseTreeExporter, _MPLTreeExporter, _DOTTreeExporter
from sklearn.tree._reingold_tilford import buchheim
from sklearn.tree import _tree
import numpy as np


def _color_brew(n):
    """Generate n colors with equally spaced hues.
    Parameters
    ----------
    n : int
        The number of colors required.
    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360. / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [(c, x, 0),
               (x, c, 0),
               (0, c, x),
               (0, x, c),
               (x, 0, c),
               (c, 0, x),
               (c, x, 0)]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))),
               (int(255 * (g + m))),
               (int(255 * (b + m)))]
        color_list.append(rgb)

    return color_list


class _PolicyTreeMPLExporter(_MPLTreeExporter):
    """
    Mixin that supports writing out the nodes of a policy tree

    Parameters
    ----------
    treatment_names : list of strings, optional, default None
        The names of the two treatments
    """

    def __init__(self, *args, treatment_names=None, **kwargs):
        self.treatment_names = treatment_names
        super().__init__(*args, **kwargs)

        self.node_dict = None

    def get_fill_color(self, tree, node_id):
        # TODO. Create our own color pallete for multiple treatments. The one below is for binary treatments.
        # Fetch appropriate color for node
        if 'rgb' not in self.colors:
            self.colors['rgb'] = _color_brew(tree.n_outputs)  # [(179, 108, 96), (81, 157, 96)]

        node_val = tree.value[node_id][:, 0]
        node_val = node_val - np.min(node_val)
        if np.max(node_val) > 0:
            node_val = node_val / np.max(node_val)
        return self.get_color(node_val)

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
        if self.node_dict is not None:
            return self._node_replacement_text_with_dict(tree, node_id, criterion)
        value = tree.value[node_id][:, 0]
        node_string = 'value = %s' % np.round(value[1:] - value[0], self.precision)

        if tree.children_left[node_id] == _tree.TREE_LEAF:
            node_string += self.characters[4]
            # Write node mean CATE
            node_string += 'Treatment = '
            if self.treatment_names:
                class_name = self.treatment_names[np.argmax(value)]
            else:
                class_name = "T%s%s%s" % (self.characters[1],
                                          np.argmax(value),
                                          self.characters[2])
            node_string += class_name

        return node_string

    def _node_replacement_text_with_dict(self, tree, node_id, criterion):

        # Write node mean CATE
        node_info = self.node_dict[node_id]
        node_string = 'CATE' + self.characters[4]
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
            else:
                raise ValueError("can only handle up to 1d values")
        else:
            value_text += "{}".format(np.around(mean, self.precision))
            if 'ci' in node_info:
                value_text += " ({}, {})".format(np.around(node_info['ci'][0], self.precision),
                                                 np.around(node_info['ci'][1], self.precision))
            value_text += self.characters[4]
        node_string += value_text

        if tree.children_left[node_id] == _tree.TREE_LEAF:
            # Write recommended treatment and value - cost
            value = tree.value[node_id][:, 0]
            node_string += 'value - cost = %s' % np.round(value[1:], self.precision) + self.characters[4]

            value = tree.value[node_id][:, 0]
            node_string += "Treatment: "
            if self.treatment_names:
                class_name = self.treatment_names[np.argmax(value)]
            else:
                class_name = "T%s%s%s" % (self.characters[1],
                                          np.argmax(value),
                                          self.characters[2])
            node_string += "{}".format(class_name)
            node_string += self.characters[4]

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


class PolicyInterpreter:
    """
    Attributes
    ----------
    criterion : {'policy_reg'}, default to 'policy_reg' # TODO: may add more criterion

    n_outputs_ : int
        The number of outputs when fit() is performed.

    max_features_ : int
        The inferred value of max_features.

    n_features_in_ : int
        Number of features seen during fit().

    max_depth : int, default to None

    min_samples_split : int or float, default to 2

    min_samples_leaf : int or float, default to 1

    random_state : int

    max_leaf_nodes : int, default to None

    min_impurity_decrease : float, default to 0.0

    ccp_alpha : non-negative float, default to 0.0

    Methods
    ----------
    fit(data, outcome, treatment,
        adjustment=None, covariate=None, treat=None, control=None,)
        Fit the model on data.

    estimate(data=None, quantity=None)
        Estimate the causal effect of the treatment on the outcome in data.

    apply(v)
        Return the index of the leaf that each sample is predicted as.

    decision_path(v)
        Return the decision path.

    _prepare4est(data)
        Prepare for the estimation of the causal effect.

    Reference
    ----------
    This implementation is based on the implementation of BaseDecisionTree
    of sklearn.
    """

    def __init__(
        self, *,
        criterion='policy_reg',
        splitter='best',
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=2022,
        max_leaf_nodes=None,
        max_features=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0,
        min_weight_fraction_leaf=0.0,
    ):
        """
        Many parameters are similar to those of BaseDecisionTree of sklearn.

        Parameters
        ----------
        criterion : {'policy_reg'}, default to 'policy_reg' # TODO: may add more criterion
            The function to measure the quality of a split. The criterion for
            training the tree is (in the Einstein notation)
            
            .. math::
            
                    S = \sum_i g_{ik} y^k_{i},
                    
            where :math:`g_{ik} = \phi(v_i)_k` is a map from the covariates, :math:`v_i`, to a
            basis vector which has only one nonzero element in the :math:`R^k` space. By
            using this criterion, the aim of the model is to find the index of the
            treatment which will render the max causal effect, i.e., finding the
            optimal policy.

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
            - If float, then `min_samples_leaf` is a fraction and
            `ceil(min_samples_leaf * n_samples)` are the minimum
            number of samples for each node.

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
        self._is_fitted = False

        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features

        self.node_dict_ = None


    def fit(
        self,
        data,
        est_model,
        *,
        covariate=None,
        effect=None,
        effect_array=None,
    ):
        """Fit the PolicyInterpreter model to interpret the policy for the causal
        effect estimated by the est_model on data.

        Parameters
        ----------
        data : pandas.DataFrame
            The input samples for the est_model to estimate the causal effects
            and for the CEInterpreter to fit.

        est_model : estimator_model
            est_model should be any valid estimator model of ylearn which was 
            already fitted and can estimate the CATE.
        """
        from ylearn.policy.policy_model import PolicyTree

        covariate = est_model.covariate if covariate is None else covariate
        # outcome = est_model.outcome

        assert covariate is not None, 'Need covariate to interpret the causal effect.'

        if est_model is not None:
            assert est_model._is_fitted
        
        self.covariate = covariate

        # y = convert2array(data, outcome)[0]
        # n, _y_d = y.shape
        # assert _y_d == 1

        self._tree_model = PolicyTree(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
            max_features=self.max_features,
            min_impurity_decrease=self.min_impurity_decrease,
            ccp_alpha=self.ccp_alpha,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
        )

        self._tree_model.fit(
            data=data,
            covariate=covariate,
            effect=effect,
            effect_array=effect_array,
            est_model=est_model
        )

        self._is_fitted = True

        return self

    def interpret(self, data=None):
        assert self._is_fitted, 'The model is not fitted yet. Please use the fit method first.'

        v = self._tree_model._check_features(v=None, data=data)

        policy_pred = self._tree_model.predict_opt_effect(data=data)
        policy_value = policy_pred.max(1)
        policy_ind = policy_pred.argmax(1)
        node_indicator = self._tree_model.decision_path(v=v)
        leaf_id = self._tree_model.apply(v=v)
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

            result_dict[f'sample_{sample_id}'] += f'The recommended policy is treatment {policy_ind[sample_id]} with value {policy_value[sample_id]}'
            
        return result_dict

    def plot(
        self, *,
        feature_names=None,
        treatment_names=None,
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
        """Plot the tree model.
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

        if feature_names == None:
            feature_names = self.covariate

        #
        exporter = _PolicyTreeMPLExporter(feature_names=feature_names, treatment_names=treatment_names,
                                           max_depth=max_depth,
                                           filled=filled,
                                           rounded=rounded, precision=precision, fontsize=fontsize)

        exporter.export(self._tree_model, ax=ax)
