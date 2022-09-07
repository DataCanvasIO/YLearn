import abc
import re

import numpy as np
from sklearn.tree import _tree
from sklearn.tree._export import _BaseTreeExporter
from sklearn.tree._reingold_tilford import buchheim, Tree
from sklearn.tree import _criterion
from numbers import Integral


class TreeProxy(object):
    """Attach node_dict attr for any tree object"""

    def __init__(self, tree, nodes_ext):
        self.tree = tree
        self.nodes_ext = nodes_ext

    def __getattribute__(self, name):
        if name == 'nodes_ext':
            return object.__getattribute__(self, "nodes_ext")
        else:
            self_tree = object.__getattribute__(self, "tree")
            return object.__getattribute__(self_tree, name)


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


# MPL
class _TreeExporter(_BaseTreeExporter):
    """
    Fix sklearn convert color bug and add text label on edge of decision tree
    """
    def __init__(
        self,
        max_depth=None,
        feature_names=None,
        treatment_names=None,
        class_names=None,
        label="all",
        filled=False,
        impurity=True,
        node_ids=False,
        proportion=False,
        rounded=False,
        precision=3,
        fontsize=None,
        include_uncertainty=False,
        uncertainty_level=0.1,
    ):

        super().__init__(
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
        )
        self.fontsize = fontsize

        # validate
        if isinstance(precision, Integral):
            if precision < 0:
                raise ValueError(
                    "'precision' should be greater or equal to 0."
                    " Got {} instead.".format(precision)
                )
        else:
            raise ValueError(
                "'precision' should be an integer. Got {} instead.".format(
                    type(precision)
                )
            )

        # The depth of each node for plotting with 'leaf' option
        self.ranks = {"leaves": []}
        # The colors to render each node with
        self.colors = {"bounds": None}

        self.characters = ["#", "[", "]", "<=", "\n", "", ""]
        self.bbox_args = dict()
        if self.rounded:
            self.bbox_args["boxstyle"] = "round"

        self.arrow_args = dict(arrowstyle="<-")

        self.include_uncertainty = include_uncertainty
        self.uncertainty_level = uncertainty_level
        self.treatment_names = treatment_names

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
        return "#%02x%02x%02x" % tuple(color)  # Fix color format error

    def _make_tree(self, node_id, et, criterion, nodes_ext,  depth=0):
        # traverses _tree.Tree recursively, builds intermediate
        # "_reingold_tilford.Tree" object
        name = self.node_to_str(TreeProxy(et, nodes_ext=nodes_ext), node_id, criterion=criterion)
        if et.children_left[node_id] != _tree.TREE_LEAF and (
            self.max_depth is None or depth <= self.max_depth
        ):
            children = [
                self._make_tree(
                    et.children_left[node_id], et, criterion, depth=depth + 1, nodes_ext=nodes_ext
                ),
                self._make_tree(
                    et.children_right[node_id], et, criterion, depth=depth + 1, nodes_ext=nodes_ext
                ),
            ]
        else:
            return Tree(name, node_id)
        return Tree(name, node_id, *children)

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

    def export(self, decision_tree, node_dict, ax=None):

        import matplotlib.pyplot as plt
        from matplotlib.text import Annotation
        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_axis_off()
        my_tree = self._make_tree(0, decision_tree.tree_, decision_tree.criterion, nodes_ext=node_dict)
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
        self.recurse(draw_tree, decision_tree.tree_, ax, max_x, max_y, text_pos=0)   # update inoke recurse

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

    @abc.abstractmethod
    def node_replacement_text(self, tree, node_id, criterion):
        raise NotImplemented

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


class _CateTreeExporter(_TreeExporter):

    def __init__(self, include_uncertainty=False, uncertainty_level=0.1,
                 *args, treatment_names=None, **kwargs):
        self.include_uncertainty = include_uncertainty
        self.uncertainty_level = uncertainty_level
        self.treatment_names = treatment_names
        super().__init__(*args, **kwargs)

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

    def node_replacement_text(self, tree, node_id, criterion):
        # Write node mean CATE
        node_info = tree.nodes_ext[node_id]
        node_string = '------CATE mean------' + self.characters[4]
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
        node_string += "------CATE std------" + self.characters[4]
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


class _PolicyTreeMPLExporter(_TreeExporter):

    def __init__(self, *args, show_all_treatments=True, **kwargs):
        self.show_all_treatments = show_all_treatments
        super().__init__(*args, **kwargs)

        self.node_dict = None

    def get_fill_color(self, tree, node_id):
        if 'rgb' not in self.colors:
            self.colors['rgb'] = _color_brew(tree.n_outputs)  # [(179, 108, 96), (81, 157, 96)]

        node_val = tree.value[node_id][:, 0]
        node_val = node_val - np.min(node_val)
        if np.max(node_val) > 0:
            node_val = node_val / np.max(node_val)
        return self.get_color(node_val)

    def ensure_treatments(self, value):
        if self.treatment_names is not None:
            return self.treatment_names
        else:
            return ["T%s%s%s" % (self.characters[1], i,self.characters[2]) for i in range(len(value))]

    def node_replacement_text(self, tree, node_id, criterion):
        # NOTE does not calc node_dict yet
        # if self.node_dict is not None:
        #     return self._node_replacement_text_with_dict(tree, node_id, criterion)
        value = tree.value[node_id][:, 0]

        node_string = ""
        if not self.show_all_treatments:
            node_string = 'value = %s' % np.round(value[1:] - value[0], self.precision)

        # if tree.children_left[node_id] == _tree.TREE_LEAF:  # NOTE: for all node not only leaf

        treatments = self.ensure_treatments(value)
        treatments_str = "------CATE mean------\n"
        if self.show_all_treatments:
            for k, v in zip(treatments, value):
                treatments_str += f"{k}={np.round(v, self.precision)}\n"

        else:
            treatments_str += "Treatment: "
            if self.treatment_names:
                # f"{}={}"
                treatments_str += self.treatment_names[np.argmax(value)]
            else:
                treatments_str += "T%s%s%s" % (self.characters[1],
                                               np.argmax(value),
                                               self.characters[2])
        node_string += treatments_str

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
            treatments = self.ensure_treatments(value)

            if self.show_all_treatments:
                treatments_str = ""
                for k, v in zip(treatments, value):
                    treatments_str += f"{k}={v}"

            else:
                treatments_str = "Treatment: "
                if self.treatment_names:
                    # f"{}={}"
                    treatments_str += self.treatment_names[np.argmax(value)]
                else:
                    treatments_str += "T%s%s%s" % (self.characters[1],
                                              np.argmax(value),
                                              self.characters[2])

            node_string += treatments_str
            node_string += self.characters[4]

        return node_string
