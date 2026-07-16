# 行政执法案卷智能审查

## 外部知识

上下文无关审查事项可以通过 `外部知识` 数组声明可插拔知识函数。知识函数
统一接收案件 metadata、目录、当前规则和 section OCR，但自行决定实际使用
哪些输入。函数返回空列表时，审查提示词不会生成空的
`external_knowledge` 区块。

当前知识函数、注册方式和索引维护命令见
[`src/external_knowledge/README.md`](src/external_knowledge/README.md)。
