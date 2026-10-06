# 知汇 AI · 通用知识检索平台

一个可直接运行的本地 RAG Web 平台，适用于企业制度、合同、报告、说明书、项目资料等不同类型的知识库。

## 快速启动

首次运行先安装依赖：

```powershell
cd knowledge-rag-ui
python -m pip install -r requirements.txt
```

配置模型后启动：

```powershell
$env:LOCAL_MODEL_PATH = "D:\A_model\Kimi-VL-A3B-Instruct"
$env:HKPC_LLM_API_KEY = "你的在线 API Key"
python main.py
```

程序默认启动 `http://127.0.0.1:8000` 并自动打开浏览器。使用 `python main.py --no-browser` 可以只启动服务。

## 知识库文档

默认知识库目录：

```text
knowledge_base/moldpdf/
```

支持 PDF、Word（`.docx`）、Excel（`.xlsx`、`.xlsm`）、PowerPoint（`.pptx`）、图片（`.png`、`.jpg`、`.jpeg`、`.bmp`、`.webp`、`.tif`、`.tiff`）、`.txt` 和 `.md`。图片会先经 OCR 提取文字。可以直接把文件复制到该目录，也可以在网页中点击“知识库”或“文档管理”上传、查看和删除资料。文档发生变化后，向量索引会自动重建。

若要使用其他位置：

```powershell
$env:KNOWLEDGE_BASE_DIR = "D:\my-knowledge-base"
```

## 模型选择

- 本地：`Kimi-VL-A3B-Instruct`，问题及资料不发送到云端。
- HKPC GPU Platform 云模型：`qwen3.8-27b`、`deepseek-v4-flash-w8a8-mtp` 和 `minimax-m3`，通过 HKPC OpenAI 兼容接口调用；API 请求会使用对应的 `public/` 平台模型 ID，默认选中 `qwen3.8-27b`。

RAG 默认走 HKPC 在线向量接口：`public/qwen3-embedding-0.6b` 建索引，`public/qwen3-reranker-8b` 对召回片段重排序。未配置 `HKPC_LLM_API_KEY` 时回退到本机 `Qwen/Qwen3-Embedding-0.6B` 和 `Qwen/Qwen3-Reranker-8B`。可用 `HKPC_VECTOR_BACKEND=local` 强制本机，或用 `HKPC_EMBEDDING_MODEL`、`HKPC_RERANKER_MODEL` 覆盖模型名称。在线检索会把文档片段发送到 HKPC。模型按需懒加载，用户的选择会保存在浏览器中。

FAISS 索引默认保存在项目目录 `faiss/`。本目录路径含中文时，FAISS 会经系统临时目录中转后再写回 `faiss/`。可用 `FAISS_INDEX_DIR` 覆盖索引位置。

## 切块与索引时间

上传资料会先按格式解析，再由 `zhihui_chunking.py` 进行结构化切块：

- PDF、Word、TXT、Markdown、PowerPoint：优先按标题、段落和中文标点切分，每块最多 800 字，保留最多 120 字重叠，并保留页码或投影片信息。
- Excel：按工作表处理，表格每 20 行为一块并重复表头，避免查询结果脱离列名。
- 图片：OCR 文字作为独立图片块，保留原始文件来源。

纯文本切块非常快：本机基准为 36 万字符生成 455 块约 `0.007` 秒。实际等待时间主要取决于 PDF/OCR 解析和 HKPC embedding 调用。当前这批注塑资料（7 份文档、1024 个片段）用 HKPC 在线 embedding 首次建索引约 20 秒；完成后缓存在项目 `faiss/`，资料未改动时后续启动只需几秒加载。可用 `HKPC_VECTOR_BACKEND=local` 改回本机向量化。

## 版权

Copyright © 2026 Hong Kong Productivity Council (HKPC). All rights reserved.

## 主要文件

```text
main.py             单一启动入口
api_server.py       文档检索、模型路由与管理 API
index.html          网页结构
app.js              网页交互
styles.css          网页样式
requirements.txt    Python 依赖
knowledge_base/     本地知识库资料
faiss/              本地 FAISS 向量索引
```

在线 API Key 只通过 `HKPC_LLM_API_KEY` 环境变量读取，不要写入网页或代码。
