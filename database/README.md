# database 目录说明

该目录存放审查服务运行时使用的本地检索数据。它不是独立数据库，不需要启动
MySQL、PostgreSQL 或向量数据库服务，前端也不需要直接读取其中的文件。

## 目录内容

```text
database/
├─ longhua_subdistrict_penalty_items/
│  ├─ items.json
│  ├─ embeddings.npy
│  └─ manifest.json
└─ shenzhen_city_management_discretion/
   ├─ violation_embeddings.npy
   └─ violation_embeddings_manifest.json
```

- `longhua_subdistrict_penalty_items`：龙华区街道承接行政处罚事项的目录和检索
  索引，用于辅助判断街道是否具有相应执法权限。
- `shenzhen_city_management_discretion`：深圳市城市管理行政处罚裁量标准的语义
  排序索引，用于在检索增强结果中向人工展示相关裁量标准。

这些小型索引随项目代码一起发布。通过 Git 克隆仓库或获取项目发布包后即可
得到，不需要单独下载。

裁量标准的结构化正文位于：

```text
knowledge/深圳市城市管理行政处罚裁量权实施标准/
  shenzhen_city_management_discretion_2023.json
```

该运行必需文件同样随项目发布；其他原始知识材料不进入 Git。

## 前端开发者

前端不需要安装模型，也不需要访问 `database`。审查结果和检索增强结果均由
后端接口返回。

## 后端开发者

后端首次使用检索功能时需要向量模型 `BAAI/bge-small-zh-v1.5`：

- 运行环境可以访问互联网：无需额外操作，首次检索时会自动下载模型并缓存。
- 运行环境不能访问互联网：先在可联网机器的项目根目录执行：

```powershell
conda run -n myenv python src/tools/download_retrieval_model.py
```

然后将生成的目录完整复制到离线部署包的相同位置：

```text
database/longhua_subdistrict_penalty_items/model/
```

模型目录体积较大，不随 Git 仓库发布。如果部署时缺少模型且运行环境不能联网，
相关向量检索会被跳过，不影响其他审查流程运行。

模型来源：<https://huggingface.co/BAAI/bge-small-zh-v1.5>，许可证为 MIT。
