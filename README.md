# Gemini API Proxy (Docker Edition)

这是一个高性能的 Google Gemini API 代理服务，支持 OpenAI 格式兼容、多密钥轮询/并发加速、流式输出以及安全设置自动注入。

**核心功能**：
- **OpenAI 接口兼容**：支持 `/v1/chat/completions` 等标准端点。
- **密钥池管理**：支持 **Polling (轮询)** 和 **Concurrent (并发竞速)** 两种模式，突破速率限制。
- **自动解除审查**：自动注入 `safetySettings`，防止 Google 过度拒绝回答。
- **Docker 部署**：一键启动，配置热更新。

---

## 速部署 (Docker)

确保您的服务器已安装 [Docker](https://docs.docker.com/get-docker/) 和 Docker Compose。

### 1. 克隆仓库
将代码下载到服务器：
```bash
git clone https://github.com/zgojin/gemini_deploy.git
cd gemini_deploy
```
### 2. ⚙️ 配置密钥 (重要)
**必须修改配置文件**，填入您的真实 Key。
- `custom_api_keys`: 设置给客户端使用的密码（自定义任意字符串，建议以 `sk-` 开头以获得最佳兼容性）。
- `native_api_keys`: 填入真实的 Google Gemini API Keys。
- `request_mode`:
  - `polling`: 轮询模式（按顺序使用 Key，省配额）。
  - `concurrent`: 并发模式（同时发起请求，速度最快）。

### 3.  启动服务
```bash
docker compose up -d --build
