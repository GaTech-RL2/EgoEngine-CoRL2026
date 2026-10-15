# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types


def test_imports():
    import spider
    from spider.simulators import dexmachina

    assert isinstance(spider.ROOT, str)
    assert isinstance(dexmachina, types.ModuleType)
