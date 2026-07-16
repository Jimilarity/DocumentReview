# 中立知识检索核心

`knowledge_retrieval` 提供独立于消费方式的事实提取、目录查询和向量排序能力。
它既不知道审查提示词，也不知道最终结果文件格式。

依赖方向固定为：

```text
knowledge_retrieval
  ↑                    ↑
external_knowledge   human_support
```

禁止 `knowledge_retrieval` 反向导入 `external_knowledge` 或 `human_support`。

## 目录结构

```text
knowledge_retrieval/
  case_routing.py
  common/
    legal_citations.py
  city_management_discretion/
    paths.py
    models.py
    catalog.py
    reranker.py
    routing.py
```

### `common/legal_citations.py`

负责从任意文书 OCR 中一次提取：

- 直接规定行政处罚后果的法律、法规、规章名称及条、款、项、目；
- 文书实际作出的处罚种类、对象、金额和原文。

模型输出受严格 JSON Schema 约束，不由程序正则修正自然语言。相同 OCR 和提取
范围使用内存缓存。

### `city_management_discretion/catalog.py`

加载完整的 2023 年版城市管理裁量标准，并提供：

- 法规全称与条号精确倒排检索；
- 法规是否被目录覆盖；
- 条号是否被目录覆盖；
- 完整裁量记录读取；
- 违法行为源数据指纹。

正式知识文件仍位于：

```text
knowledge/深圳市城市管理行政处罚裁量权实施标准/
  shenzhen_city_management_discretion_2023.json
```

### `city_management_discretion/reranker.py`

在精确法条候选内，根据案由与违法行为计算向量相似度并排序。向量派生文件仍
位于：

```text
database/shenzhen_city_management_discretion/
  violation_embeddings.npy
  violation_embeddings_manifest.json
```

重排器只返回完整 `ranked`，不包含 Top 1 阈值、自动选择或裁量判断策略。

### `case_routing.py`

集中维护“综行罚、应急罚、消防罚、市监罚”等最低限度案号路由。外部知识和检索
增强不得分别复制正则。

## 数据构建

生成或更新裁量法条检索元数据：

```powershell
conda run -n myenv python src/tools/build_city_management_discretion_catalog.py
```

重建违法行为向量：

```powershell
conda run -n myenv python src/tools/build_city_management_discretion_embeddings.py
```

查看检索增强真实候选：

```powershell
conda run -n myenv python src/tools/inspect_city_management_discretion_retrieval.py `
  --law-name "《深圳经济特区市容和环境卫生管理条例》" `
  --article 65 `
  --case-reason "某垃圾转运站未安装视频监控设备案"
```

旧的裁量适用性模型、外部知识注入函数以及相应测试脚本已经删除。处罚是否落入
某个裁量分组由人工根据 `retrieval_enhancement_results.json` 判断。
