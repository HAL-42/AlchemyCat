#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: tracker.py
@time: 2020/2/26 13:40
@desc:
"""
import json
import os
import pickle
from pprint import pformat, pprint
from typing import Any, Callable, DefaultDict, Optional
from collections import defaultdict, OrderedDict
import warnings

from alchemy_cat.py_tools import indent
from alchemy_cat.py_tools.numpy_json_encoder import NumpyArrayEncoder

__all__ = ['Statistic', 'Tracker', 'OneOffTracker']


class _StatisticStatue(object):

    def __init__(self):
        self.last_update_count: int = -1
        self.last_statistic: Any = None


def _check_importance(importance):
    if not isinstance(importance, int) or importance < 0 or importance > 9:
            raise ValueError(f"importance {importance} must >= 0 and <= 9")


class Statistic(object):
    """Decorator for statistic of Tracker

    Function decorated should specify how to calculate the statistic from Tracker object. The Statistic obj can
    caching the last statistic and return it if Tracker is not updated since last calculate.

    You can also use Statistic.getter to create statistics with importance, which can be filtered when get
    self.statistics
    """

    def __init__(self, fget: Callable, importance: int=0):
        self.fget = fget
        self.name = fget.__name__
        self.__doc__ = fget.__doc__

        _check_importance(importance)
        self.importance = importance

    @classmethod
    def getter(cls, importance: int=0):
        """Decorate statics with importance"""
        _check_importance(importance)

        def decorator(fget):
            return cls(fget, importance)

        return decorator

    def __get__(self, obj: 'Tracker', obj_type=None):
        if obj is None:
            return self

        statistic_statue: _StatisticStatue = obj._statistic_status[self.name]

        if statistic_statue.last_update_count < obj._update_count:
            # * Calculate statistic
            statistic_statue.last_update_count = obj._update_count
            statistic_statue.last_statistic = self.fget(obj)
            return statistic_statue.last_statistic
        elif statistic_statue.last_update_count > obj._update_count:
            # * Raise Error
            raise RuntimeError(f"Statistic {self.name}'s last_update_count {statistic_statue.last_update_count} can't "
                               f"larger than tracker {obj}'s update_count {obj.update_count}")
        else:
            # * Return cache
            return statistic_statue.last_statistic

    def __repr__(self):
        return f"Statistic <{self.__class__}>: {self.name}"


class Tracker(object):
    """A tracker to some progress.

    Tracker object is used to tracking some progress. It can be updated with progress and calculate statistic
    of the tracked progress.
    """

    def __init__(self, init_dict: Optional[OrderedDict]=None):
        """Init tracker's attributes

        Must be called before first in custom __init__.

        Args:
            init_dict: init_dict for reset the tracker. Make sure to call by super().__init__(quick_init(self, locals())
                if subclass is directly driven from Tracker, other wise call
                Tracker.__init__(self, quick_init(self, locals()) in last driven class's __init__ or just modify
                self.init_dict If None, then can't use self.reset.
        """
        self._update_count: int = 0
        self._statistic_status: DefaultDict[str, _StatisticStatue] = defaultdict(_StatisticStatue)
        self.init_dict = init_dict

    def update(self, *args, **kwargs):
        """Update tracker's statues

        Must be called first in custom update
        """
        if hasattr(self, 'is_one_off') and self.is_one_off and not self.is_in_one_off_block:
            warnings.warn(f"You are Updating the one-off tracker {self} out-off the with block", RuntimeWarning)
        self._update_count += 1

    @property
    def update_count(self):
        """Return update count of current tracker"""
        return self._update_count

    def statistics(self, importance: int=0) -> OrderedDict:
        """Return all statistic of current tracker in the dict {statistic.name: statistic}

        Args:
            importance (int): importance filter. Only statistics with (statistics.importance > importance) will be
                returned
        """
        _check_importance(importance)

        tracker_classes = [base_cls for base_cls in self.__class__.mro() if issubclass(base_cls, Tracker)]

        statistics = []
        statistic_names = []
        for cls in tracker_classes:
            for statistic in cls.__dict__.values():
                # Make sure that on class's statistic can override base class's statistic
                if isinstance(statistic, Statistic) and statistic.name not in statistic_names:
                    statistics.append(statistic)
                    statistic_names.append(statistic.name)

        statistic_dict = {statistic: statistic.__get__(self, self.__class__)
                for statistic in statistics if statistic.importance >= importance}

        sorted_statistics = sorted(statistic_dict.items(), key=lambda item: str(item[0].importance) + item[0].name)

        ordered_statistic = OrderedDict()
        for k, v in sorted_statistics:
            ordered_statistic[k.name] = v

        return ordered_statistic

    def save_statistics(self, save_dir: str, importance: int=0):
        """Save statistics needing saving to save_dir.

        If self.statistic can be saved by json, then function will save statistics to statistics.json, else to
        statistics.txt. self.statistics will be pickle to statistic.pkl anyway.

        Args:
            save_dir: Dictionary to save the statistic.json/txt and statistic.pkl
            importance (int): importance filter. Only statistics with statistics.importance > importance will be saved
        """
        _check_importance(importance)
        os.makedirs(save_dir, exist_ok=True)

        # * Save statistics
        try:
            with open(os.path.join(save_dir, 'statistics.json'), 'w') as json_f:
                json.dump(self.statistics(importance), json_f, indent=4, cls=NumpyArrayEncoder)
        finally:
            with open(os.path.join(save_dir, 'statistics.txt'), 'w') as txt_f:
                txt_f.write(pformat(self.statistics(importance), indent=4))

            with open(os.path.join(save_dir, 'statistics.pkl'), 'wb') as pkl_f:
                pickle.dump(self.statistics(importance), pkl_f)

    def print_statistics(self, importance: int=0):
        """Print statistics filtered by importance

        Args:
            importance: importance filter. Only statistics with statistics.importance > importance will be saved

        Returns:
            Formatted statistics filtered by importance
        """
        _check_importance(importance)

        pprint(self.statistics(importance))
        return pformat(self.statistics(importance), indent=4)

    def reset(self):
        if self.init_dict is None:
            raise RuntimeError(f"Tracker {self} has no init_dict. Set init_dict with quick_init in __init__")

        return self.__init__(**self.init_dict)

    def __repr__(self):
        return f"Tracker <{self.__class__}>: update_count = {self.update_count}"

    def __str__(self):
        return pformat(self.statistics(), indent=4)


class OneOffTracker(object):
    """An context manager which to make sure that tracker can only be updated within the context"""

    def __init__(self, tracker_factory: Callable[[], Tracker]):
        """An context manager which to make sure that tracker can only be updated within the context

        Args:
            tracker_factory: Return the one-off tracker
        """
        self._tracker = tracker_factory()
        self._tracker.is_one_off = True
        self._tracker.is_in_one_off_block = False

    def __enter__(self):
        self._tracker.is_in_one_off_block = True
        return self._tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tracker.is_in_one_off_block = False

    def __repr__(self):
        return f"OneOffTracker <{self.__class__}>: \n" \
            + indent(f"tracker: {self._tracker}\n") \
            + indent(f"tracker.is_in_one_off_block: {self._tracker.is_in_one_off_block}")


# TODO: Trigger decorator which can be trigger after function called several times.
