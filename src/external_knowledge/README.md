# 外部知识提示词注入

`external_knowledge` 只负责把已经确认适合交给审查模型的知识转换为
`KnowledgeItem(content)`，并按规则中具体文书审查事项的 `外部知识` 列表注入
`<external_knowledge>`。

它不再承载通用法条提取、裁量目录加载、向量检索等共享能力。共享检索能力统一
位于 [`knowledge_retrieval`](../knowledge_retrieval/README.md)。

## 当前知识函数

| 名称 | 状态 | 说明 |
| --- | --- | --- |
| `case_directory_info` | 已实现 | 将完整案卷目录作为知识提供给明确配置的审查事项 |
| `longhua_subdistrict_penalty_items_catalog` | 已实现 | 检索龙华区街道承接行政处罚事项 |
| `legal_citation_validity` | 待实现 | 预留法条时效性 API |
| `shenzhen_administrative_litigation_jurisdiction` | 待实现 | 预留深圳行政诉讼管辖知识 |

城市管理裁量标准不再注册为外部知识函数。规则 360 使用
`检索增强: ["city_management_discretion_candidates"]`，将检索结果输出给人工，
不注入普通审查提示词。

## 目录职责

```text
external_knowledge/
  models.py
  registry.py
  service.py
  functions.py
  case_directory/
  longhua_subdistrict_penalty/
  legal_citation_validity/
  administrative_litigation_jurisdiction/
```

- `models.py`：提示词注入使用的最小协议；
- `registry.py`：稳定名称注册表；
- `service.py`：并发调用知识函数、过滤空结果；
- `functions.py`：默认知识函数注册入口；
- 每个具体知识函数放在独立目录。

## 规则配置

```json
{
  "审查事项": "核查执法权限。",
  "评查说明": "",
  "外部知识": [
    "longhua_subdistrict_penalty_items_catalog"
  ]
}
```

没有配置或列表为空时，不生成 `<external_knowledge>`。知识函数返回空列表时也不
生成空标签。

## 扩展约束

新增知识函数时，只在适配器中完成以下工作：

1. 从统一 `KnowledgeContext` 选择需要的输入；
2. 调用本地文件、数据库、API 或 `knowledge_retrieval` 中的共享检索能力；
3. 将确认适合主审查模型使用的内容转换为 `KnowledgeItem`。

不要在该目录复制通用检索、向量索引或事实提取实现。配置了未注册名称时视为
配置错误；可选知识源运行失败时记录警告并跳过，不中断普通审查。
