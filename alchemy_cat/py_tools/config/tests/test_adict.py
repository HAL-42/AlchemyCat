#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2024/6/24 19:12
@File    : test_adict.py
@Software: PyCharm
@Desc    : 
"""
import pickle
from copy import deepcopy

import pytest

from alchemy_cat.py_tools import ADict

__all__ = ['d', 'ref_d', 'dic', 'ref_dic', 'to_update', 'dic_can_exec']


@pytest.fixture(scope="function")  # 字典可能被修改，所以每次都要重新生成。
def d() -> dict:
    yield {'num': 0,
           'lst': [1, [2, {'lst_lst_dic': 'exit'}], {'lst_dic': 'exit'}],
           'dic': {'sub_num': 3, 'sub_set': frozenset((1, 2, 3))},
           'uni_dic': {'sub_num': 4},
           'obj': object()}


@pytest.fixture(scope="function")
def ref_d(d) -> dict:
    d['lst'][1].append(d['lst'])
    d['inner_lst'] = d['lst'][1]
    d['lst'].append(d)
    d['uni_dic']['prev_dic'] = d['dic']
    yield d


@pytest.fixture(scope="function")
def dic(d) -> ADict:
    yield ADict(d)


@pytest.fixture(scope="function")
def ref_dic(ref_d) -> ADict:
    yield ADict(ref_d)

# -* 测试拷贝时自引用。


def test_ref_dic(ref_dic: ADict):
    assert ref_dic['lst'][1][-1] is ref_dic['lst']
    assert ref_dic['inner_lst'] is ref_dic['lst'][1]
    assert ref_dic['lst'][-1] is ref_dic
    assert ref_dic['uni_dic']['prev_dic'] is ref_dic['dic']


def test_ref_dic_copy(ref_dic: ADict):
    copy_ref_dic = ref_dic.copy()
    assert copy_ref_dic['lst'][1][-1] is copy_ref_dic['lst']
    assert copy_ref_dic['inner_lst'] is copy_ref_dic['lst'][1]
    assert copy_ref_dic['lst'][-1] is copy_ref_dic
    assert copy_ref_dic['uni_dic']['prev_dic'] is copy_ref_dic['dic']


def test_deepcopy_inner_outer_ref(ref_dic: ADict):
    ref_dic_lst = [ref_dic, ref_dic, ref_dic]
    nl = deepcopy(ref_dic_lst)
    assert nl[0] is nl[1] is nl[2] is not ref_dic_lst[0]
    assert nl[0]['lst'][1][-1] is nl[0]['lst']
    assert nl[0]['inner_lst'] is nl[0]['lst'][1]
    assert nl[0]['lst'][-1] is nl[0]
    assert nl[0]['uni_dic']['prev_dic'] is nl[0]['dic']


def test_pickle_ref_dic(ref_dic: ADict):
    nd = pickle.loads(pickle.dumps(ref_dic))
    assert nd['lst'][1][-1] is nd['lst']
    assert nd['inner_lst'] is nd['lst'][1]
    assert nd['lst'][-1] is nd
    assert nd['uni_dic']['prev_dic'] is nd['dic']


def test_todict_ref(ref_dic: ADict):
    to_dic = ref_dic.to_dict()
    assert to_dic['lst'][1][-1] is to_dic['lst']
    assert to_dic['inner_lst'] is to_dic['lst'][1]
    assert to_dic['lst'][-1] is to_dic
    assert to_dic['uni_dic']['prev_dic'] is to_dic['dic']


def test_pickle_remain_whole_frozen(dic: ADict):
    dic.uni_dic.set_whole()
    dic.freeze()
    nd = pickle.loads(pickle.dumps(dic))
    assert nd.uni_dic.is_whole
    assert nd.is_frozen
    assert nd.dic.is_frozen
    assert nd.uni_dic.is_frozen


def test_branches_dict_ref(ref_dic: ADict):
    b = list(ref_dic.branches)
    assert len(b) == 3
    assert b[0] is ref_dic
    assert b[1] is ref_dic.dic
    assert b[2] is ref_dic.uni_dic


def test_ckl_dict_ref(ref_dic: ADict):
    ns, cs, ks, ls = [], [], [], []
    for name, (c, k, l) in ref_dic.named_ckl:
        ns.append(name)
        cs.append(c)
        ks.append(k)
        ls.append(l)

    assert len(ns) == len(cs) == len(ks) == len(ls) == 7

    assert ls[0] == 0
    assert ls[1] is ref_dic.lst
    assert ls[2] == 3
    assert ls[3] is ref_dic.dic.sub_set
    assert ls[4] == 4
    assert ls[5] is ref_dic.obj
    assert ls[6] is ref_dic.inner_lst

    assert ns[0] == 'num'
    assert ns[1] == 'lst'
    assert ns[2] == 'dic.sub_num'
    assert ns[3] == 'dic.sub_set'
    assert ns[4] == 'uni_dic.sub_num'
    assert ns[5] == 'obj'
    assert ns[6] == 'inner_lst'

    assert cs[0] is ref_dic
    assert cs[1] is ref_dic
    assert cs[2] is ref_dic.dic
    assert cs[3] is ref_dic.dic
    assert cs[4] is ref_dic.uni_dic
    assert cs[5] is ref_dic
    assert cs[6] is ref_dic

    assert ks[0] == 'num'
    assert ks[1] == 'lst'
    assert ks[2] == 'sub_num'
    assert ks[3] == 'sub_set'
    assert ks[4] == 'sub_num'
    assert ks[5] == 'obj'
    assert ks[6] == 'inner_lst'


# -* 测试dict_update。


@pytest.fixture(scope="function")
def to_update() -> ADict:
    to_update = ADict()
    to_update.my_num = -1
    to_update.lst = 'lst'  # 人有我有，都是value。
    to_update.dic.sub_num = 30  # 人有我有，都是子树 -> 人有我有，都是value。
    to_update.dic.sub_tuple = (10,)  # 人有我有，都是子树 -> 人无我有。
    yield to_update


def test_update(to_update: ADict, dic: ADict):
    to_update.update(dic.freeze(), dic={'add_by_kwargs': 8}, new_key='lala')
    assert to_update == {'my_num': -1,
                         'lst': [1, [2, {'lst_lst_dic': 'exit'}], {'lst_dic': 'exit'}],
                         'dic': {'sub_num': 3, 'sub_tuple': (10,), 'sub_set': frozenset({1, 2, 3}), 'add_by_kwargs': 8},
                         'num': 0,
                         'uni_dic': {'sub_num': 4},
                         'obj': to_update.obj,
                         'new_key': 'lala'}
    assert to_update.obj is dic.obj
    assert to_update.lst is not dic.lst
    assert to_update.uni_dic is not dic.uni_dic


def test_update_whole(to_update: ADict, dic: ADict):
    dic = dic.freeze()
    dic.dic.set_whole()
    to_update.update(dic)
    assert to_update == {'my_num': -1,
                         'lst': [1, [2, {'lst_lst_dic': 'exit'}], {'lst_dic': 'exit'}],
                         'dic': {'sub_num': 3, 'sub_set': frozenset((1, 2, 3))},
                         'num': 0,
                         'uni_dic': {'sub_num': 4},
                         'obj': to_update.obj}


def test_update_incremental(to_update: ADict, dic: ADict):
    to_update.dict_update(dic.freeze(), incremental=True)
    assert to_update == {'my_num': -1,
                         'lst': 'lst',
                         'dic': {'sub_num': 30, 'sub_tuple': (10,), 'sub_set': frozenset((1, 2, 3))},
                         'num': 0,
                         'uni_dic': {'sub_num': 4},
                         'obj': to_update.obj}


def test_update_or(to_update: ADict, dic: ADict):
    new_dic = to_update | dic.freeze()
    assert new_dic == {'my_num': -1,
                       'lst': [1, [2, {'lst_lst_dic': 'exit'}], {'lst_dic': 'exit'}],
                       'dic': {'sub_num': 3, 'sub_tuple': (10,), 'sub_set': frozenset({1, 2, 3})},
                       'num': 0,
                       'uni_dic': {'sub_num': 4},
                       'obj': dic.obj}
    assert new_dic is not to_update


def test_update_ror(to_update: ADict, dic: ADict):
    new_dic = dic.to_dict() | to_update
    assert new_dic == {'my_num': -1,
                       'lst': [1, [2, {'lst_lst_dic': 'exit'}], {'lst_dic': 'exit'}],
                       'dic': {'sub_num': 3, 'sub_tuple': (10,), 'sub_set': frozenset({1, 2, 3})},
                       'num': 0,
                       'uni_dic': {'sub_num': 4},
                       'obj': dic.obj}
    assert new_dic is not to_update


def test_update_ior(to_update: ADict, dic: ADict):
    ori_id = id(to_update)
    to_update |= dic.freeze()
    assert to_update == {'my_num': -1,
                         'lst': [1, [2, {'lst_lst_dic': 'exit'}], {'lst_dic': 'exit'}],
                         'dic': {'sub_num': 3, 'sub_tuple': (10,), 'sub_set': frozenset({1, 2, 3})},
                         'num': 0,
                         'uni_dic': {'sub_num': 4},
                         'obj': dic.obj}
    assert id(to_update) == ori_id

# -* 测试freeze功能。


def test_freeze(dic: ADict):
    dic.freeze()
    assert dic.is_frozen
    assert dic.dic.is_frozen
    assert dic.uni_dic.is_frozen

    with pytest.raises(RuntimeError):
        dic.new_key = 1

    with pytest.raises(KeyError):
        _ = dic.new_key

    with pytest.raises(RuntimeError):
        del dic.dic.sub_num

    with pytest.raises(RuntimeError):
        dic.clear()

    with pytest.raises(RuntimeError):
        dic.pop('new_key')

    with pytest.raises(RuntimeError):
        dic.popitem()

    with pytest.raises(RuntimeError):
        dic.dic.empty_leaf()

    dic.unfreeze()
    assert not dic.is_frozen
    assert not dic.dic.is_frozen
    assert not dic.uni_dic.is_frozen


# -* 测试重写的受影响的字典方法。

def test_dic_from_keys(dic: ADict):
    value = [99, 88, 77]
    dic_from_keys = ADict.fromkeys(['a', 'b', 'c'], [99, 88, 77])
    assert dic_from_keys == {'a': value, 'b': value, 'c': value}
    assert dic_from_keys['a'] is dic_from_keys['b'] is dic_from_keys['c']
    assert dic_from_keys['a'] is not value


# -* 测试set_whole。对其update的效果再update测试中已经完成。

def test_set_whole(dic: ADict):
    dic.dic.set_whole()
    assert dic.dic.is_whole

    dic.new.new_new.new_new_new.set_whole()
    assert 'new_new_new' in dic.new.new_new
    assert dic.new.new_new.new_new_new.is_whole

    dic.uni_dic.empty_leaf()
    assert dic.uni_dic.is_whole
    assert len(dic.uni_dic) == 0

    dic.dic.empty_tree()
    assert not dic.dic.is_whole
    assert len(dic.dic) == 0

    dic.new_empty.new_new_empty.empty_leaf()
    assert dic.new_empty.new_new_empty.is_whole
    assert len(dic.new_empty.new_new_empty) == 0

    dic.new_tree.new_new_tree.empty_tree()
    assert not dic.new_tree.new_new_tree.is_whole
    assert len(dic.new_tree.new_new_tree) == 0


# -* 测试to_txt()

@pytest.fixture(scope="function")
def dic_can_exec(dic: ADict) -> ADict:
    del dic.obj
    dic.uni_dic.set_whole()
    yield dic


def test_to_txt(dic_can_exec: ADict):
    exec(dic_can_exec.to_txt('exec_dic'), globals(), ldict := {})
    exec_dic = ldict['exec_dic']
    assert exec_dic == dic_can_exec
    assert exec_dic.uni_dic.is_whole
