# 数据集导出/载入（新增功能）

这个脚本只新增新文件，不改旧代码。

## 1. 导出（默认命名为时间）

```bash
python3 reasearch/dataset_tool.py export
```

导出后的目录：

`exports/<时间>/data/`（原始 CSV）
`exports/<时间>/model/`（训练模型）

## 2. 导出（自定义名字）

```bash
python3 reasearch/dataset_tool.py export --name my_dataset
```

## 3. 导出指定用户（profiles）

```bash
python3 reasearch/dataset_tool.py export --profile player1
```

## 4. 查看已有导出

```bash
python3 reasearch/dataset_tool.py list
```

## 5. 导入（载入）

```bash
python3 reasearch/dataset_tool.py import my_dataset
```

## 6. 导入到指定用户

```bash
python3 reasearch/dataset_tool.py import my_dataset --profile player1
```
