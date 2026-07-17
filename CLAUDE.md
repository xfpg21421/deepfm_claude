# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

本项目用于完整复现论文：`paper/deepfm.pdf`
（DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, Guo et al., 2017）

## 目标

- 忠实复现论文
- 不臆造论文没有提到的方法
- 所有实现必须能够追溯到论文
- 所有代码必须可以运行
- 所有模块必须有测试

## 工作方式

整个项目采用 TODO 驱动。Claude 必须：

1. 打开 `TODO.md`
2. 找到第一个未完成任务
3. 完成该任务
4. 更新 `TODO.md`
5. 提交当前成果
6. 自动继续下一项

除非遇到以下情况，否则不要停止：

- 信息不足
- 论文存在歧义
- 测试无法通过
- 需要用户决策

## 实现规范

基于 TensorFlow 来实现。任何代码必须：

- 有类型注解
- 有 Docstring
- 有 pytest
- 解释 Tensor Shape
- 不允许 placeholder

## Review 规范

每完成一个模块必须：

- 自查代码
- 与论文逐项比对
- 修复发现的问题

Review 完成后才能继续。

## 测试规范

任何模块：`pytest` 必须通过。否则禁止继续。

## Git

每完成一个 TODO，自动建议：

```
git add .
git commit
```

commit message 使用：

```
完成：<TODO名称>
```

## 输出要求

不要一次生成整个工程。每次只完成一个 TODO。完成后更新 `TODO.md`。
