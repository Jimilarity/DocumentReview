# 检索增强与人工复核信息链路

检索增强与普通审查并列执行，但它不判断规则是否合格，不生成 `issues`，也不进入
`merge_rule_results`、审查结果后处理或 `review_results.json`。它只把文书事实和
知识检索候选整理到独立的 `retrieval_enhancement_results.json`，交由人工判断。

## 规则配置

配置位于规则顶层，而不是某个审查事务内部。同一规则仍可保留普通审查事项：

```json
{
  "序号": 360,
  "上下文无关审查事项": {
    "行政处罚决定书": {
      "审查事项": "……",
      "评查说明": "……",
      "外部知识": []
    }
  },
  "上下文相关审查事项": [],
  "备注": "简易程序",
  "检索增强": [
    "city_management_discretion_candidates"
  ]
}
```

`外部知识` 和 `检索增强` 是两条互不替代的协议：

- `外部知识` 的结果会进入普通审查提示词；
- `检索增强` 的结果只写入人工复核信息文件；
- 同一规则可以同时参加普通审查和检索增强；
- 规则 360 当前不再把裁量标准注入普通审查模型。

`检索增强` 未配置或为空列表时不执行。列表是否为空已经完整表达是否启用，
因此没有额外的 `启用` 字段。

## 执行粒度与一次提取

规则在规则层面筛选，实际工作按 `document_name + section_id` 分组。同一 section
被多个规则或多个模块使用时，执行器先合并所有模块的事实需求，再让同一个事实
提供器执行一次。一个提供器可以同时提供多个字段，例如处罚依据和文书实际处罚
由同一次结构化模型调用产生。

模块接口只声明所需事实并执行检索：

```python
class ExampleModule:
    name = "example"

    def required_facts(self, context) -> set[str]:
        return {"penalty_legal_citations", "imposed_penalties"}

    async def retrieve(self, context, facts):
        ...
```

新增模块后，将实例注册到 `HumanSupportModuleRegistry`。模块输出只固定一层薄信封：

```json
{
  "knowledge_name": "example",
  "status": "matched",
  "payload": {}
}
```

`payload` 由模块自行定义，避免用一套僵硬协议限制不同知识来源。`status` 可为：

- `matched`：存在检索候选；
- `no_match`：事实提取成功，但知识源没有候选；
- `skipped`：知识源不适用，或文书没有可供检索的事实；
- `error`：事实提取、配置或检索发生技术错误。

只有确实存在警告时才增加 `warnings` 字段，避免在正常结果中重复输出空数组。

## 规则 360：城市管理裁量标准候选

模块名为 `city_management_discretion_candidates`。当前执行链路：

1. 先按案号判断是否为街道综合行政执法案卷；其他案卷直接 `skipped`，不调用
   事实提取模型。
2. 从当前文书 OCR 一次提取处罚依据和文书实际处罚；案由优先读取 `meta_info`。
3. 每条处罚依据先按法规全称和条号精确检索
   《深圳市城市管理行政处罚裁量权实施标准（2023年版）》。
4. 只要有案由，每个精确候选都会计算违法行为向量相似度；即使只有一个候选也
   计算。多候选按相似度降序排列。
5. 最多返回前 5 项候选，每项包含违法行为、设定依据、完整裁量分组和相似度。

检索增强链路不会使用 Top 1 阈值删除其余候选，也不会调用裁量幅度适用性判断
模型。处罚是否落入某个裁量分组由人工结合文书事实判断。

法条协议、裁量目录和向量重排实现位于中立的
[`knowledge_retrieval`](../knowledge_retrieval/README.md)；本模块只负责案卷范围、
事实需求、Top K 和结果组装。

可以不运行完整审查，直接检查真实目录和向量索引的候选输出：

```powershell
conda run -n myenv python src/tools/inspect_city_management_discretion_retrieval.py `
  --law-name "《深圳经济特区市容和环境卫生管理条例》" `
  --article 65 `
  --case-reason "某垃圾转运站未安装视频监控设备案"
```

如果要同时观察展示给人工的文书实际处罚，可增加
`--penalty-target`、`--penalty-amount` 和 `--penalty-text`；处罚会出现在
`extracted_facts` 中，不会在检索模块 payload 中重复，也不会触发模型判断处罚
落入哪个裁量区间。

规则 360 的输出示意：

```json
{
  "rule_index": 360,
  "documents": [
    {
      "document_name": "行政处罚决定书",
      "section_id": 12,
      "extracted_facts": {
        "penalty_legal_citations": [],
        "imposed_penalties": [],
        "case_reason": "……"
      },
      "retrievals": []
    }
  ]
}
```

完整规则不重复写入结果文件；`rule_index` 足以回查 `all_rules.json`。模块名称
已经包含在每一项 `retrievals[].knowledge_name` 中，因此顶层也不再重复列出。
