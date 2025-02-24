#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2025/02/24 16:35
@File    : json_patch.py
@Software: Visual Studio Code
@Desc    :
"""
import io
import json
import typing as t

from loguru import logger

__all__ = ["JSONPatcher"]


class JSONPatcher:
    """
    A utility class that patches json.dumps and json.dump to handle non-serializable objects.

    Attributes:
        custom_serializer: Callable that converts non-serializable objects to JSON-serializable types
        ensure_ascii: Whether to escape non-ASCII characters in the output
    """

    # 保存原始的 json.dumps 和 json.dump
    _original_dumps = json.dumps
    _original_dump = json.dump

    def __init__(
        self,
        custom_serializer: t.Optional[t.Callable[[t.Any], str | int | float | bool | None | dict | list]] = None,
        ensure_ascii: bool = False,
    ):
        super().__init__()
        self.custom_serializer = custom_serializer if custom_serializer is not None else self.default_serializer
        self.ensure_ascii = ensure_ascii
        logger.info(
            (
                f"JSONPatcher is initialized with custom_serializer: {self.custom_serializer}, "
                f"ensure_ascii: {self.ensure_ascii}"
            )
        )
        self.patch_json_dump()

    @staticmethod
    def default_serializer(obj):
        # 对于无法序列化的对象，返回其字符串表示
        return str(obj)

    # 重写 json.dumps
    def patched_dumps(self, obj, **kwargs):
        if "default" not in kwargs:
            kwargs["default"] = self.custom_serializer
        if "ensure_ascii" not in kwargs:
            kwargs["ensure_ascii"] = self.ensure_ascii
        return type(self)._original_dumps(obj, **kwargs)

    # 重写 json.dump
    def patched_dump(self, obj, fp, **kwargs):
        if "default" not in kwargs:
            kwargs["default"] = self.custom_serializer
        if "ensure_ascii" not in kwargs:
            kwargs["ensure_ascii"] = self.ensure_ascii
        return type(self)._original_dump(obj, fp, **kwargs)

    def patch_json_dump(self):
        json.dumps = self.patched_dumps
        json.dump = self.patched_dump

    @classmethod
    def restore_json_dump(cls):
        """Restore original json dump functions."""
        json.dumps = cls._original_dumps
        json.dump = cls._original_dump

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore_json_dump()


if __name__ == "__main__":
    # 示例使用
    data = {
        "name": "中文",
        "age": 30,
        "custom_object1": set([1, 2, 3]),  # JSON 不支持 set 类型
        "custom_object2": JSONPatcher,
    }

    with JSONPatcher(ensure_ascii=False):
        # 直接使用 json.dumps
        json_string = json.dumps(data, indent=4)
        print(json_string)

        # 使用json.dump
        with io.StringIO() as f:
            json.dump(data, f, indent=4)
            print(f.getvalue())
