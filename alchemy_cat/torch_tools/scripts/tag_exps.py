#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2022/2/19 17:25
@File    : tag_exps.py
@Software: PyCharm
@Desc    : 根据git提交中的配置文件，给指定提交打上标签，名之以实验标识。
"""
import os
import os.path as osp
import argparse
import subprocess

from alchemy_cat.py_tools import gprint, yprint

"""
 六言·参数含义
   乾隆再世
 在指定仓库下，
 列出提交更改。
 在配置目录下，
 找到配置文件。
 文件对应实验，
 标到指定提交。
"""
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--repo', default='.', help="Git repo root dir. ", type=str)
parser.add_argument('-s', '--show', default='HEAD', help="Commit to show file changed. ", type=str)
parser.add_argument('-c', '--config_dir', default='configs', help="Relative path to repo dir of config dir. ", type=str)
parser.add_argument('-n', '--config_name', default='cfg.py', help="Config files' name. ", type=str)
parser.add_argument('-a', '--at', default='HEAD', help="Tag experiment ID at which commit. ", type=str)
parser.add_argument('-k', '--skip_tagged', default=0, help="If true, auto skip exp already tagged. ", type=int)
args = parser.parse_args()


def removeprefix(text: str, prefix: str) -> str:
    """Alternative for str.removeprefix in python 3.9.

    See Also:
        https://docs.python.org/3/library/stdtypes.html#str.removeprefix
    """
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


# * 在指定仓库下，列出某提交更改的文件。
p = subprocess.run(['git', 'show', '--pretty=', '--name-only', args.show],
                   cwd=osp.normpath(args.repo),
                   capture_output=True, encoding='utf-8', check=False)
if p.stderr:
    raise RuntimeError(f"Failed to show filed changed at commit {args.show}, git command returns: \n"
                       f"{p.stderr}")

changed_files = [osp.normpath(file) for file in p.stdout.splitlines()]

print(f"In repo {osp.abspath(args.repo)}, find following changed files at commit {args.show}: \n" +
      '\n'.join(changed_files), end="\n\n")

# * 基于更改文件，在配置目录下，找到配置文件。
config_dir = osp.normpath(args.config_dir) + os.sep
config_files = [file for file in changed_files if file.startswith(config_dir)] \
    if config_dir != '.' + os.sep else changed_files
config_files = [file for file in config_files if osp.basename(file) == args.config_name]

print(f"Find following config files in config dir {config_dir} named {args.config_name}: \n" +
      "\n".join(config_files), end="\n\n")

# * 基于配置文件，生成实验标识。
exp_ids = []
for file in config_files:
    file_wo_prefix = removeprefix(file, config_dir) if config_dir != '.' + os.sep else file
    exp_dir = osp.dirname(file_wo_prefix)
    if not exp_dir:
        raise RuntimeError(f"Config file {file} is not in exp dir. ")
    exp_ids.append(exp_dir.replace(os.sep, '-'))

# * 获取仓库已有标签。
p = subprocess.run(['git', 'tag'],
                   cwd=osp.normpath(args.repo),
                   capture_output=True, encoding='utf-8', check=False)
if p.stderr:
    raise RuntimeError(f"Failed to list tags at repo {osp.abspath(args.repo)}, git command returns: \n"
                       f"{p.stderr}")

tags = p.stdout.splitlines()

# * 对找到的每个配置文件及对应实验标识，由用户决定，是否要在指定提交打标签。
for i, (config_file, exp_id) in enumerate(zip(config_files, exp_ids)):
    if args.skip_tagged and (exp_id in tags):
        yprint(f"[{i+1}/{len(config_files)}] Skip {exp_id} for {config_file}, for it was already tagged. ", end="\n\n")
        continue

    while True:
        selection = input(f"For {config_file}, tag {exp_id} at commit {args.at}?[y/n]: ")
        if selection == 'y':
            p = subprocess.run(['git', 'tag', exp_id, args.at],
                               cwd=osp.normpath(args.repo),
                               capture_output=True, encoding='utf-8', check=False)
            if p.stderr:
                raise RuntimeError(f"Failed to tag {exp_id} at commit {args.at}, git command returns: \n"
                                   f"{p.stderr}")
            gprint(f"[{i+1}/{len(config_files)}] Tag {exp_id} at commit {args.at}. ", end="\n\n")
            break
        elif selection == 'n':
            if exp_id not in tags:
                yprint(f"[{i+1}/{len(config_files)}] Don't tag {exp_id} at commit {args.at}, while "
                       f"this id is still not tagged. ", end="\n\n")
            else:
                yprint(f"[{i+1}/{len(config_files)}] Don't tag {exp_id} at commit {args.at}, attention "
                       f"this id is already tagged. Use -k 1 to avoid re-tag. ", end="\n\n")
            break
        else:
            print(f"Unknown selection {selection}, please re-enter. ")
