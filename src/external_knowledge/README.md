# 外部知识提示词注入

`external_knowledge` 只负责把已经确认适合交给审查模型的知识转换为
`KnowledgeItem(content)`，并按规则中具体文书审查事项或上下文相关审查事项的
`外部知识` 列表注入 `<external_knowledge>`。

它不再承载通用法条提取、裁量目录加载、向量检索等共享能力。共享检索能力统一
位于 [`knowledge_retrieval`](../knowledge_retrieval/README.md)。

## 当前知识函数

| 名称 | 状态 | 说明 |
| --- | --- | --- |
| `case_directory_info` | 已实现 | 将完整案卷目录作为知识提供给明确配置的审查事项 |
| `longhua_subdistrict_penalty_items_catalog` | 已实现 | 检索龙华区街道承接行政处罚事项 |
| `legal_citation_validity` | 已实现 | 调用版本化法条检索 API，供模型核对法条和时效 |
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

法条检索固定使用预处理阶段生成的全案 `案由` 和 `案情`。同一文书审查项可增加可选
`法条检索查询`，其中 `字段` 用于增补已写入案卷元数据的具体字段；空数组或省略时仅
使用案由和案情。`日期字段`按配置从案卷元数据取值：

```json
{
  "外部知识": ["legal_citation_validity"],
  "法条检索查询": {
    "字段": [],
    "日期字段": ["案发日期", "违法行为发生日期"],
    "top_k": 3,
    "version_k": 7
  }
}
```

案情在全文 OCR 完成后统一提取并写入 `meta_info.json`，与封面提取的案由共同作为
每次检索的必备内容。调用内容最多 300 个字符；`日期字段`中第一个有效的 `YYYY-MM-DD` 日期传为
`as_of`。公网服务地址、密钥和调用参数通过项目根目录 `.env` 配置；密钥不得写入
规则文件、源代码或文档：

```dotenv
LAW_RETRIEVAL_BASE_URL=https://review.zfqp.fun/law-api
LAW_RETRIEVAL_API_KEY=在此填写已获授权的密钥
LAW_RETRIEVAL_TIMEOUT_SECONDS=60
LAW_RETRIEVAL_MAX_RETRIES=2
```

未设置 `LAW_RETRIEVAL_BASE_URL` 时默认使用上述公网地址。每个请求以
`X-API-Key` 发送密钥；仅对 `502`、`503` 最多重试两次。候选法条明确标为核对资料，
不能自动确定违法依据或处罚依据。

没有配置或列表为空时，不生成 `<external_knowledge>`。知识函数返回空列表时也不
生成空标签。

上下文无关审查在对应文书项配置 `外部知识`；上下文相关审查在对应任务项配置
`外部知识`。两类审查共用同一注册表和调用机制。上下文相关审查会将该任务已提取的
字段值与案卷元数据一并提供给知识函数，因此法条检索的可选 `字段` 可以直接填写该
任务 `字段` 中的字段名。

上下文相关审查事项同样可配置 `外部知识`；其结果与配置的跨文书字段来源一同交给
对应任务。例如规则 113 使用 `case_directory_info` 核对是否存在负责人集体讨论笔录。

## 扩展约束

新增知识函数时，只在适配器中完成以下工作：

1. 从统一 `KnowledgeContext` 选择需要的输入；
2. 调用本地文件、数据库、API 或 `knowledge_retrieval` 中的共享检索能力；
3. 将确认适合主审查模型使用的内容转换为 `KnowledgeItem`。

不要在该目录复制通用检索、向量索引或事实提取实现。配置了未注册名称时视为
配置错误；可选知识源运行失败时记录警告并跳过，不中断普通审查。
