#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/23 22:17
@File    : cache_dir.py
@Software: PyCharm
@Desc    : 
"""
import os
import os.path as osp
import subprocess
import uuid

from .str_formatters import get_local_time_str

__all__ = ['CacheDir']


class CacheDir(os.PathLike):

    def __init__(self,
                 save_at: str | os.PathLike='/dev/null',
                 cache_dir: str | os.PathLike='/tmp/cache_dir',
                 exist: str='error',
                 save_when_del: bool=False,
                 save_when_exit: bool=False,
                 enabled: bool=True):
        """将save_at下的文件缓存到cache_at下。

        Args:
            save_at: 最终保存目录。
            cache_dir: 缓存目录的父目录，推荐/tmp/xxx、/shm/xxx。
            exist: 'error' - 如果save_at已经存在，则报错；'cache': 将save_at转移到cache_at；'delete': 删除save_at。
            save_when_del: 是否在CacheDir被del时dump。
            save_when_exit: 是否在程序退出时dump。
            enabled: 是否启用CacheDir，如果不启用，初始化后，就处于_saved状态。此时等价于save_at。
        """
        super().__init__()
        save_at, cache_dir = str(save_at), str(cache_dir)

        self.save_dir = osp.dirname(save_at)
        self.save_at = save_at

        self.cache_dir = cache_dir
        self.cache_at = osp.join(self.cache_dir, f'{get_local_time_str(for_file_name=True)}@{uuid.uuid4().hex}')

        assert exist in ['error', 'cache', 'delete']
        self.exist = exist

        self.save_when_del = save_when_del
        self.save_when_exit = save_when_exit
        if self.save_when_exit:
            raise NotImplementedError(f"{type(self)}: dump_when_exit功能尚未实现。")

        self.enabled = enabled

        if enabled:
            self._saved: bool | str = False
            self._cache()
        else:
            os.makedirs(self.save_at, exist_ok=True)
            self._saved: bool | str = '未启用'

    @property
    def saved(self):
        return self._saved

    @property
    def at(self):
        if self.saved:
            return self.save_at
        else:
            return self.cache_at

    def __str__(self):
        return self.at

    def __fspath__(self):
        return self.at

    def __repr__(self):
        return f"{type(self)}: (exist: {self.exist}, saved: {self.saved})\n" \
               f"   save_at: {self.save_at} \n" \
               f"   cache_at: {self.cache_at} \n"

    def _cache(self):
        """将save_at下的文件缓存到cache_at下"""
        if self._saved:
            raise RuntimeError(f"{type(self)}: CacheDir状态为{self.saved}，不能再次cache。")

        # * 新建cache_at。
        os.makedirs(self.cache_at, exist_ok=False)

        # * 若save_at存在，则根据exist进行处理。
        if osp.isdir(self.save_at):
            match self.exist:
                case 'error':
                    raise RuntimeError(f"{type(self)}: {self.save_at} 已经存在，不能cache。")
                case 'cache':
                    print(f"{type(self)}: {self.save_at} 已经存在，正在将其转移到 {self.cache_at} ...")
                    subprocess.run(['cp', '-a', osp.join(self.save_at, ''), self.cache_at])
                case 'delete':
                    print(f"{type(self)}: {self.save_at} 已经存在，正在删除...")
                    subprocess.run(['rm', '-r', self.save_at])
                case _:
                    raise RuntimeError(f"{type(self)}: 不支持的exist参数。")

    def flush(self, size_only: bool=False):
        """将cache_at下的文件同步到save_at下

        Args:
            size_only: rsync时，只比较size。
        """
        if self._saved:
            print(f"{type(self)}:  CacheDir状态为{self.saved}，flush无效。")
            return

        if self.save_at == '/dev/null':
            print(f"{type(self)}: save_at为/dev/null，flush无效。")
            return

        # * 确保save_at的目录存在。
        os.makedirs(self.save_at, exist_ok=True)

        # * 将cache_at用rsync同步到save_at。
        print(f"{type(self)}: 正在将 {self.cache_at} 同步到 {self.save_at} ...")
        if size_only:
            subprocess.run(['rsync', '-a', '--size-only', '--delete', osp.join(self.cache_at, ''), self.save_at])
        else:
            subprocess.run(['rsync', '-a', '--delete', osp.join(self.cache_at, ''), self.save_at])

    def _rm_cache(self):
        print(f"{type(self)}: 正在删除 {self.cache_at} ...")
        subprocess.run(['rm', '-r', self.cache_at])

    def save(self, size_only: bool=False, no_flush: bool=False):
        """将cache_at下的文件转移到save_at下

        Args:
            size_only: rsync时，只比较size。
            no_flush: 如果能肯定cache和save已经一致，无需进行flush，可以设置no_flush=True。
        """
        if self._saved:
            print(f"{type(self)}: CacheDir状态为{self.saved}，save无效。")
            return

        print(f"{type(self)}: 正在将 {self.cache_at} 保存到 {self.save_at} ...")

        # * 进行最后一次同步。
        if not no_flush:
            self.flush(size_only=size_only)

        # * 删除cache_at。
        self._rm_cache()

        self._saved = '已保存'

    def terminate(self):
        """直接退出，删除cache_at。"""
        if self._saved:
            print(f"{type(self)}: CacheDir状态为{self.saved}，terminate无效。")
            return

        print(f"{type(self)}: 正在终止...")

        # * 删除cache_at。
        self._rm_cache()

        self._saved = '已终止'

    def __del__(self):
        if self.save_when_del:
            self.save()
