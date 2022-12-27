import sys as sys_

is_os_windows = sys_.platform.find('win') == 0
is_os_darwin = sys_.platform.find('darwin') == 0
is_os_linux = sys_.platform.find('linux') == 0

try:
    from packaging.version import Version
except ModuleNotFoundError:
    from distutils.version import LooseVersion as Version

from ._common import const, set_random_state, infer_task_type, unique, to_repr
from ._common import to_df, context, is_notebook, view_pydot
from ._common import to_list, join_list
from ._common import to_snake_case, to_camel_case, drop_none
from ._common import check_fitted, check_fitted_
