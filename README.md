# Qwen Image Studio

一个自托管的图像生成 Web 应用。后端把 Hugging Face Space 的
`/generate` 接口封装成任务队列和 HTTP API，前端提供生图、参考图上传、
历史画廊、管理员控制台和访问密钥管理。

> 本项目与 Qwen、Hugging Face 无隶属关系，也不包含模型权重。使用时请遵守
> 上游模型、Space 和服务商的使用条款及配额限制。

## 功能

- 文生图、参考图编辑、透明 PNG 三种模式
- 最多 10 张参考图，支持原生多图 Space 和单图 Space 自动拼图兜底
- FastAPI 任务队列、进度查询、取消和失败退款
- 卡密式访问密钥：次数、有效期、停用和并发安全扣减
- 管理员控制台：密钥、账号池、代理池、渠道和画廊
- Hugging Face Token 池与代理池，可运行时增删和冷却
- 单文件 CSS、内联 SVG，无前端构建步骤
- Docker 镜像和 Compose 部署

## 目录

| 路径 | 作用 |
|---|---|
| `server.py` | FastAPI 服务、任务队列、鉴权和管理接口 |
| `qwen_client.py` | Space 客户端、参数校验、账号池和代理池 |
| `auth.py` | 访问密钥、HMAC 会话和管理员口令 |
| `channels.py` | Space 渠道探测与配置 |
| `materials.py` | 用户素材索引 |
| `index.html` / `admin.html` / `login.html` | 前端页面 |
| `static/theme.css` | 统一主题 |
| `hf_space_multi/` | 支持最多 10 张参考图的 Hugging Face Space |
| `config/` | 配置示例，不含真实凭据 |

## 本地启动

需要 Python 3.11 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements-dev.txt
python server.py --host 127.0.0.1 --port 7860
```

打开 <http://127.0.0.1:7860/login>。

如果没有设置 `QWEN_ADMIN_PASSWORD`，首次启动会生成随机管理员口令，打印到
控制台并保存到运行时数据目录的 `admin_password.txt`。首次登录后请修改口令
并删除该文件。

运行时数据默认写在源码目录，也可以显式指定：

```bash
export QWEN_DATA_DIR=/var/lib/qwen-image-studio
python server.py --host 0.0.0.0 --port 7860
```

## Docker

```bash
cp .env.example .env
docker compose up --build -d
docker compose logs -f
```

镜像把运行时数据放在 `/data` 卷中。默认端口是 `7860`，可通过 `.env` 中的
`QWEN_PORT` 修改。

单独构建和运行：

```bash
docker build -t qwen-image-studio:local .
docker run --rm -p 7860:7860 \
  -e QWEN_ADMIN_PASSWORD='replace-me' \
  -v qwen-data:/data \
  qwen-image-studio:local
```

仓库附带的 GitHub Actions 工作流会在推送到 `main`、推送版本标签或手动运行时
构建镜像；非 PR 构建会发布到：

```text
ghcr.io/<github-owner>/qwen-image-studio
```

## 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `QWEN_DATA_DIR` | 源码目录 | 运行时数据根目录；Docker 中为 `/data` |
| `QWEN_ADMIN_PASSWORD` | 随机生成 | 管理员口令 |
| `QWEN_HF_TOKENS` | 空 | Token 池，格式见 `config/tokens.example.txt` |
| `QWEN_PROXIES` | 空 | 代理池，格式见 `config/proxies.example.txt` |
| `QWEN_SPACES` | 主站 | 逗号分隔的兼容 Space ID |
| `QWEN_MULTI_SPACES` | 空 | 参考图入参为列表的 Space |
| `QWEN_STRATEGY` | `proxy_first` | `proxy_first`、`token_first`、`proxy_only`、`token_only` |
| `QWEN_CONCURRENCY` | `1` | 本地任务并发数 |
| `QWEN_QUOTA_COOLDOWN_H` | 到下一个 UTC 零点 | 配额耗尽后的冷却时间 |

也可以通过运行时数据目录中的 `tokens.txt` 和 `proxies.txt` 配置账号与代理。
这两个文件已被 `.gitignore` 和 `.dockerignore` 排除。

## 运行时数据

以下文件由服务生成，均不应提交到 Git：

```text
data/
├── .admin.json
├── .session.key
├── admin_password.txt
├── channels.json
├── gallery.json
├── jobs.json
├── licenses.json
├── materials.json
├── proxies.txt
├── tokens.txt
├── outputs/
└── uploads/
```

`channels.json`、`gallery.json`、`jobs.json`、`licenses.json` 和
`materials.json` 包含部署状态、提示词、任务记录或访问密钥，应按敏感数据
处理。

## API

主要入口：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/login` | 登录页 |
| `GET` | `/app` | 生图页，需要登录 |
| `GET` | `/admin` | 管理控制台，需要管理员 |
| `GET` | `/api/config` | 允许的模式、画幅和参数范围 |
| `POST` | `/api/auth/login` | 管理员口令或访问密钥登录 |
| `POST` | `/api/generate` | 创建生成任务 |
| `GET` | `/api/job/{id}` | 查询任务 |
| `POST` | `/api/job/{id}/cancel` | 取消任务 |
| `POST` | `/api/upload` | 上传参考图 |
| `GET` | `/api/gallery` | 画廊 |
| `GET` | `/docs` | FastAPI OpenAPI 文档 |

管理员接口在 `/api/pool`、`/api/proxy`、`/api/licenses` 和
`/api/overview` 下。

## 测试

```bash
pytest
```

测试使用临时数据目录，不读取真实 Token、代理、图片或部署状态，也不会调用
上游 Space。

## 安全

- 公网部署前请配置 HTTPS 和反向代理。
- 不要在公网启用 `QWEN_NO_AUTH=1` 或 `--no-auth`。
- 只使用你拥有或获授权使用的账号、代理和 Space，并遵守上游条款与限流。
- 提交前运行 `git status`，确认没有运行时数据或本地 `.env`。
- 泄露 Token 后应立即在上游撤销；仅从最新提交删除并不能清除 Git 历史。

更多说明见 [SECURITY.md](SECURITY.md)。

## License

[MIT](LICENSE)
