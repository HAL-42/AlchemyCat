#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2021/8/10 17:50
@File    : voc_annotation.py
@Software: PyCharm
@Desc    :
"""
from typing import Tuple, Optional, List, Dict, Any

from collections import defaultdict
from xml.etree import ElementTree as ET
from xml.dom import minidom

__all__ = ['dump_voc_annotation', 'load_voc_annotation', 'count_obj_num']


def _sub_element_with_text(parent: ET.Element, tag: str, text: Optional[str]=None) -> ET.Element:
    tmp = ET.SubElement(parent, tag)
    if text is not None:
        tmp.text = text
    return tmp


def dump_voc_annotation(xml_file: str,
                        folder: Optional[str]=None,
                        filename: Optional[str]=None,
                        source_database: Optional[str]=None, source_annotation: Optional[str]=None,
                        source_image: Optional[str]=None,
                        size: Optional[Tuple[int, int]]=None,
                        segmented: Optional[int]=None,
                        objects: Optional[List[Dict[str, Any]]]=None) -> str:
    # * Create <annotation>
    root = ET.Element("annotation")

    # * Create <folder>
    if folder is not None:
        _sub_element_with_text(root, 'folder', folder)

    # * Create <filename>
    if filename is not None:
        _sub_element_with_text(root, 'filename', filename)

    # * Create <source>
    if source_database is not None:
        assert source_annotation is not None
        assert source_image is not None
        source_tag = _sub_element_with_text(root, 'source')
        _sub_element_with_text(source_tag, 'database', source_database)
        _sub_element_with_text(source_tag, 'annotation', source_annotation)
        _sub_element_with_text(source_tag, 'image', source_image)

    # * Create <size>
    if size is not None:
        img_h, img_w = size
        size_tag = _sub_element_with_text(root, 'size')
        _sub_element_with_text(size_tag, 'width', str(img_w))
        _sub_element_with_text(size_tag, 'height', str(img_h))
        _sub_element_with_text(size_tag, 'depth', '3')

    # * Create <segmented>
    if segmented is not None:
        _sub_element_with_text(root, 'segmented', str(segmented))

    # * Create <object>
    if objects is not None:
        for obj in objects:
            # * Get object attrib
            name = obj['name']
            pose = obj.get('pose', 'Unspecified')
            truncated = obj.get('truncated', '0')
            difficult = obj.get('difficult', '0')
            xmin, ymin, xmax, ymax = obj['bndbox']

            # * Create object tag
            object_tag = _sub_element_with_text(root, 'object')
            _sub_element_with_text(object_tag, 'name', name)
            _sub_element_with_text(object_tag, 'pose', pose)
            _sub_element_with_text(object_tag, 'truncated', str(truncated))
            _sub_element_with_text(object_tag, 'difficult', str(difficult))
            bndbox_tag = _sub_element_with_text(object_tag, 'bndbox')
            _sub_element_with_text(bndbox_tag, 'xmin', str(xmin))
            _sub_element_with_text(bndbox_tag, 'ymin', str(ymin))
            _sub_element_with_text(bndbox_tag, 'xmax', str(xmax))
            _sub_element_with_text(bndbox_tag, 'ymax', str(ymax))

    xml_str = minidom.parseString(ET.tostring(root)).childNodes[0].toprettyxml(indent='	')  # No xml header

    with open(xml_file, 'w') as f:
        f.write(xml_str)

    return xml_str


def load_voc_annotation(xml_file: str) -> Dict[str, Any]:
    def to_int(obj):
        return int(obj) if obj is not None else obj

    tree = ET.parse(xml_file)
    root = tree.getroot()

    ret = {}

    # * Load <folder>
    ret['folder'] = root.findtext('folder')

    # * Load <filename>
    ret['filename'] = root.findtext('filename')

    # * Load <source>
    ret['source_database'] = root.findtext('source/database')
    ret['source_annotation'] = root.findtext('source/annotation')
    ret['source_image'] = root.findtext('source/image')

    # * Load <size>
    ret['size'] = (to_int(root.findtext('size/height')), to_int(root.findtext('size/width')))

    # * Load <segmented>
    ret['segmented'] = to_int(root.findtext("segmented"))

    # * Load <object>
    ret['objects'] = []
    for object_tag in root.findall('object'):
        ret['objects'].append({
            'name': object_tag.findtext('name'),
            'pose': object_tag.findtext('pose'),
            'truncated': to_int(object_tag.findtext('truncated')),
            'difficult': to_int(object_tag.findtext('difficult')),
            'bndbox': [to_int(object_tag.findtext('bndbox/xmin')),
                       to_int(object_tag.findtext('bndbox/ymin')),
                       to_int(object_tag.findtext('bndbox/xmax')),
                       to_int(object_tag.findtext('bndbox/ymax'))]
        })

    return ret


def count_obj_num(anno: Dict[str, Any]) -> Dict[str, int]:
    obj_num = defaultdict(int)

    for obj in anno['objects']:
        obj_num[obj['name']] += 1

    return dict(obj_num)
