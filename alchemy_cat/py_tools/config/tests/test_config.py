#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/6/25 21:10
@File    : test_config.py
@Software: PyCharm
@Desc    : 
"""
from copy import copy

import pytest

from alchemy_cat.py_tools import Config, ADict, DEP

from .test_adict import *


@pytest.fixture(scope="function")
def cfg_can_exec(dic_can_exec):
    yield Config(dic_can_exec, caps=('a', 'b'))

# -* 测试Config的文本化。


def test_cfg_to_txt(cfg_can_exec: Config):
    exec(cfg_can_exec.to_txt('exec_cfg'), globals(), ldict := {})
    exec_cfg = ldict['exec_cfg']
    assert exec_cfg == cfg_can_exec
    assert exec_cfg.uni_dic.is_whole
    assert exec_cfg.get_attribute('_cfgs_update_at_parser') == cfg_can_exec.get_attribute('_cfgs_update_at_parser')

# -* 测试初始化。


def test_not_go_in_list(dic_can_exec: ADict, cfg_can_exec: Config):
    assert cfg_can_exec.lst is dic_can_exec.lst

# TODO 一个待拆解的巨大测试。


def test_cfg_all():
    d = {'num': 0,
         'lst': [1, [2, {'lst_lst_dic': 'exit'}], {'lst_dic': 'exit'}],
         'dic': {'sub_num': 3, 'sub_set': frozenset((1, 2, 3)), 'sub_dic': {'sub_sub_tuple': (4, 5)}},
         'uni_dic': {'sub_num': 4}}

    cfg = Config(d)
    cfg.rslt_dir = '/tmp/test_alchemy_cat_config'
    cfg.uni_dic.set_whole()

    indep_tree = Config({'a': 1, 'b': 2, 'c': {'cc': 3, 'dd': {'ddd': 4}}, 'e': {'ee': 5}})
    indep_tree.c.il_rel = DEP(lambda c: c.a - c.b, rel=True)  # rel_IL
    indep_tree.c.il_not_rel = DEP(lambda c: c.num - c.i1.i2.c.dd.ddd, rel=False)  # not_rel_IL

    cfg.d1.d2.d3.k = 'v'  # missing + set_v + set_unmounted_dic

    cfg.i1.i2 = indep_tree  # set root tree

    cfg.f1.f2 = indep_tree.e  # set subtree

    new_cfg = copy(cfg)

    cfg.load(create_rslt_dir=False)

    assert cfg == {'num': 0,
                   'lst': [1, [2, {'lst_lst_dic': 'exit'}], {'lst_dic': 'exit'}],
                   'dic': {'sub_num': 3,
                           'sub_set': frozenset({1, 2, 3}),
                           'sub_dic': {'sub_sub_tuple': (4, 5)}},
                   'uni_dic': {'sub_num': 4},
                   'rslt_dir': '/tmp/test_alchemy_cat_config',
                   'd1': {'d2': {'d3': {'k': 'v'}}},
                   'i1': {'i2': {'a': 1,
                                 'b': 2,
                                 'c': {'cc': 3, 'dd': {'ddd': 4}, 'il_rel': -1, 'il_not_rel': -4},
                                 'e': {'ee': 5}}},
                   'f1': {'f2': {'ee': 5}}}

    # 均mounted且有正确父子关系。
    bs = list(cfg.branches)
    assert len(bs) == 14

    for b in bs:
        assert b._is_mounted

    root_num = 0
    for b in bs:  # 所有branch有正确的父子关系。
        if b.__parent is None:
            assert b.__key is None
            root_num += 1
        else:
            assert b.__parent[b.__key] is b
    assert root_num == 1

    # missing生成的子树链有正确的父子关系。
    assert cfg.__parent is None
    assert cfg.__key is None
    assert cfg.d1.__parent is cfg
    assert cfg.d1.__key == 'd1'
    assert cfg.d1.d2.__parent is cfg.d1
    assert cfg.d1.d2.__key == 'd2'
    assert cfg.d1.d2.d3.__parent == cfg.d1.d2
    assert cfg.d1.d2.d3.__key == 'd3'

    # 挂载独立树，能正确变更树的父子关系。
    assert cfg.i1.i2 is indep_tree
    assert cfg.i1.i2.__parent is cfg.i1
    assert cfg.i1.i2.__key == 'i2'

    # 子树、父树的内部结果不受影响。
    assert cfg.i1.i2.c.__parent is cfg.i1.i2
    assert cfg.i1.i2.c.__key == 'c'

    assert cfg.i1.__parent is cfg
    assert cfg.i1.__key == 'i1'

    # 挂载非独立树，触发拷贝。
    assert cfg.f1.f2 is not indep_tree.e
    assert cfg.f1.f2 == indep_tree.e
    assert cfg.f1.f2.__parent is cfg.f1
    assert cfg.f1.f2.__key == 'f2'

    assert indep_tree.e.__parent is indep_tree
    assert indep_tree.e.__key == 'e'

    assert cfg.f1.__parent is cfg
    assert cfg.f1.__key == 'f1'

    # 测试branch_copy
    assert len(list(cfg.branches)) == len(list(new_cfg.branches))
    for b1, b2 in zip(cfg.branches, new_cfg.branches):  # branch数目不变，但是branch的id变了。
        assert b1 is not b2

    for b in new_cfg.branches:  # 所有branch均挂载。
        assert b._is_mounted

    root_num = 0
    for b in new_cfg.branches:  # 所有branch有正确的父子关系。
        if b.__parent is None:
            assert b.__key is None
            root_num += 1
        else:
            assert b.__parent[b.__key] is b
    assert root_num == 1

    assert len(list(cfg.leaves)) == len(list(new_cfg.leaves))
    for l1, l2 in zip(cfg.leaves, new_cfg.leaves):  # 叶子不变。DEP不受影响。
        if not isinstance(l2, DEP):
            assert l1 is l2

    assert new_cfg.i1.i2.c.il_rel.level == 1
    assert new_cfg.i1.i2.c.il_not_rel.level == float('inf')

    new_cfg.load(create_rslt_dir=False)
    assert new_cfg == cfg

    # 测试删除。
    cfg.unfreeze()

    del_dic = cfg.dic
    del cfg.dic
    assert 'dic' not in cfg
    assert del_dic.__parent is None
    assert del_dic.__key is None

    pop_uni_dic = cfg.pop('uni_dic')
    assert 'uni_dic' not in cfg
    assert pop_uni_dic.__parent is None
    assert pop_uni_dic.__key is None

    clear_d1, clear_i1, clear_f1 = cfg.d1, cfg.i1, cfg.f1
    cfg.clear()
    assert len(cfg) == 0
    assert clear_d1.__parent is None
    assert clear_d1.__key is None
    assert clear_i1.__parent is None
    assert clear_i1.__key is None
    assert clear_f1.__parent is None
    assert clear_f1.__key is None

    cfg.a.b = 1
    pop_k, pop_v = cfg.popitem()
    assert len(cfg) == 0
    assert pop_k == 'a'
    assert pop_v.__parent is None
    assert pop_v.__key is None
    assert pop_v.b == 1
