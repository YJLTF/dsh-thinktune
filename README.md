# dsh-thinktune

DeepSeek Harness（dsh）的 **Ollama 适配器插件**：让 qwen3.8 等思考模型的**思考强度**成为一个可配置、可选择的一等参数，同时原生支持**多模态图片输入**，并可在四种常见的思考控制协议之间一键切换。

适配版本：dsh `0.1.5-alpha.1`（`@deepseek-ai/dsh-llm@0.1.5-alpha.1`）、Ollama `0.33.0-rc2`。

---

## 它解决什么问题

dsh 接 Ollama 跑 qwen3.8 时，有三个痛点：

1. **思考强度不可控** —— qwen3 默认「开脑洞」，长思考拖慢响应、占用上下文；而 dsh 没有内置 Ollama 适配器，更没有把思考档位暴露给 agent 配置和模型选择器。
2. **协议碎片** —— 控制「思考与否/思考多少」在生态里有四种互不兼容的写法（Ollama 原生 `think` 参数、Qwen3 软开关、OpenAI `reasoning_effort`、vLLM 的 `chat_template_kwargs`），换一个推理栈就要改代码。
3. **图片输入缺失** —— qwen3.8 是多模态模型，通用适配器往往只做纯文本。

dsh 的请求在分发前是**深度冻结**的，插件不能改写请求——提供方专属的转换只能由 `LlmAdapter` 完成。所以本插件以标准 LLM 适配器形态实现，做三件事：

- 通过 `resolveModel()` 向 harness **上报该模型的思考档位**（`off / low / medium / high`，可自定义），harness 在发起任何提供方请求**之前**校验档位合法性，不支持的档位直接拒绝（`UNSUPPORTED_REASONING_EFFORT`），不会打到 Ollama 才报错；
- 在 `stream()` 里把选中的档位**翻译成四种控制协议之一**；
- 把 Ollama 返回的思考轨迹映射为 harness 的 `reasoning` 内容块（与正文分离呈现），并把用户消息里的图片引用转换为对应协议的图片载荷。

## 特性一览

| 特性 | 说明 |
|---|---|
| 统一思考档位 | `off / low / medium / high` 默认档位词表，支持自定义档位与 `budget` 思考预算 |
| 四种控制策略 | `native` / `soft-switch` / `reasoning-effort` / `template-kwarg`，改一行配置切换，见下文映射表 |
| 档位原生校验 | 档位集合由适配器声明、服务端提前校验；非法档位零 I/O 拒绝 |
| 默认档位 | `defaultEffort` 可为未选择档位的请求物化默认值；不配则保持提供方默认 |
| 思考与正文分离 | Ollama `message.thinking` → `reasoning-delta`，Web UI 里独立呈现思考轨迹；正文内联的 `<think>` 标签也能增量分离 |
| 多模态图片输入 | `/api/show` 的 `vision` capability 自动探测；图片经附件服务按像素/字节预算重编码后内联；超预算从最旧图片开始确定性卸载 |
| 健壮的流式传输 | 空闲超时（`TIMEOUT`）、请求中止（`ABORTED`）、HTTP 400/500/429 → 稳定错误码；NDJSON 与 SSE 双协议解析 |
| 会话卫生 | 默认剥离助手历史中的思考块（Qwen3 官方多轮建议），可选保留 |
| 凭据安全 | 配置里只写**环境变量名**（`apiKeyEnv`），永不放明文 key |

### 四种控制策略

| 档位 | `native`（默认） | `soft-switch` | `reasoning-effort` | `template-kwarg` |
|---|---|---|---|---|
| 端点 | `/api/chat` (NDJSON) | `/api/chat` (NDJSON) | `/v1/chat/completions` (SSE) | `/v1/chat/completions` (SSE) |
| 线上参数 | `think` | 消息内软开关 | `reasoning_effort` | `chat_template_kwargs` |
| `off` | `think: false` | 追加 `/no_think` | `reasoning_effort: "none"`* | `{enable_thinking: false}` |
| `low/medium/high` | `think: true`** | 追加 `/think` | `reasoning_effort: "low"/…` | `{enable_thinking: true, thinking_budget?: N}` |
| 自定义档位 | `think: true` | `/think` | 原样透传 | `{enable_thinking: true}` |

\* 由 `offSentinel` 配置，可改 `"minimal"` 或 `"omit"`（不发字段）。
\*\* `nativeLevels: true` 时改为 `think: "low"/"medium"/"high"`（gpt-oss 等支持等级的模型用；qwen3 保持 `false`）。

**怎么选**：

- 直连 Ollama 跑 qwen3 系 → `native`（默认，最稳）。
- Qwen3-2507 之前的混合思考版且端点不支持 `think` 参数 → `soft-switch`（注意：Qwen3-2507 分叉版不支持软开关）。
- 走 Ollama 的 OpenAI 兼容层，或 gpt-oss 系模型 → `reasoning-effort`。
- 把同一个适配器指向 vLLM / SGLang → `template-kwarg`（唯一能表达 thinking budget 的策略）。

### 图片输入（多模态）

- **能力自动声明**：`/api/show` 返回 `vision` capability（qwen3.8 即如此）时，适配器上报 `inputModalities: ['text', 'image']`，harness 才会在请求里保留图片；`imageCapability: yes/no` 可强制覆盖。
- **请求图预备**：图片引用经附件服务 `readImageRequest(ref, {maxPixels, maxBytes})` 拿到按预算重编码的请求版本（质量阶梯自动降档），然后按协议内联：
  - `native` / `soft-switch` → user 消息的 `images: ["<base64>", …]`；
  - `reasoning-effort` / `template-kwarg` → `content` 变为 `[{type:'text'}, {type:'image_url', image_url:{url:'data:image/png;base64,…'}}]`。
- **预算卸载**：超过 `imageMaxPerRequest`（默认 8 张）或 `imageMaxRequestBytes`（默认 32 MiB）时，最旧的图片替换为 dsh-llm 标准占位文本且不再读取字节；`imageMaxPerRequest: 0` 关闭图片输入。
- **前置条件**：组合里挂了附件服务（`ctx.attachments`，dsh 全系 profile 默认自带）。未挂载时含图请求报 `UNSUPPORTED_CONTENT`，纯文本不受影响。

## 安装

> **为什么有两种方式？** dsh 的插件加载器用 Node 的 TypeScript 类型剥离加载 `.ts` 插件，而 Node **拒绝处理 `node_modules` 目录下的 TS 文件**——如果你看到 `ERR_UNSUPPORTED_NODE_MODULES_TYPE_STRIPPING`，就是把 `src/index.ts` 安装进了 profile 的 `node_modules`。因此：包安装走编译好的 `dist/index.js` 入口（方式 A），只有 `node_modules` 之外的源码直连才用 TS（方式 B）。

### 方式 A：包安装（推荐）

仓库已提交 `dist/` 构建产物，安装无需本地工具链；若自己改了 `src/`，先 `pnpm install && pnpm build` 重新生成。

1. 克隆或下载本仓库到本地任意位置：
   ```sh
   git clone https://github.com/YJLTF/dsh-thinktune.git
   ```
2. 把它装进 profile（二选一）：
   ```sh
   # dsh 插件管理
   dsh plugin --profile web add F:/path/to/dsh-thinktune
   ```
   或手动编辑 `$DSH_HOME/profiles/web/package.json` 后在该目录 `pnpm install`：
   ```json
   {
     "dependencies": {
       "dsh-thinktune": "file:F:/path/to/dsh-thinktune"
     }
   }
   ```
3. 在 profile 的 `cordis.patch.yml` 里用**包名**引用（会解析到 `dist/index.js`）：
   ```yaml
   - insert:
       - id: llm-thinktune
         name: 'dsh-thinktune'
         config:
           providers:
             - ollama
           endpoint: 'http://127.0.0.1:11434'
           strategy: native
   ```

### 方式 B：源码直连（开发模式）

`name` 写绝对路径指向 `src/index.ts`（在 `node_modules` 之外，类型剥离可用）：

```yaml
- insert:
    - id: llm-thinktune
      name: 'F:/project/dsh-thinktune/src/index.ts'
      config: { /* 同上 */ }
```

需要先在插件目录 `pnpm install`。改完源码重启/HMR 即生效，无需构建；跑测试用 `pnpm test`。

### 指定默认模型与思考档位（两种方式都需要）

profile patch 把默认模型指过去：

```yaml
- id: agent-default-model
  config:
    provider: ollama
    model: qwen3.8:27b
```

再在 `$DSH_HOME/settings.yaml` 设置档位（与 Web UI 模型选择器落盘的是同一个位置）：

```yaml
agent-default-model:
  provider: ollama
  model: qwen3.8:27b
  reasoningEffort: low        # off | low | medium | high
```

> 注意：`settings.yaml` 的 `agent-default-model` 节会**覆盖** profile patch 里的同名插件配置，而 `reasoningEffort` 只能通过这个设置节表达（组合配置的 schema 只收 `provider/model`）。

重启 dsh 即生效：启动日志会有一行 `thinktune: provider route(s) ollama registered (strategy native, efforts off/low/medium/high)`。

## 使用

**切换思考强度（三选一）**：

1. **Web UI 模型选择器** —— 选中 `ollama / qwen3.8:27b` 后，模型旁会出现档位下拉（数据来自本适配器的 `resolveModel`，选择结果自动写回 `settings.yaml`）；
2. **settings.yaml** —— 改 `agent-default-model.reasoningEffort`，保存即热生效；
3. **会话内** —— 子代理或单次会话可单独指定模型与档位，互不影响默认值。

**切换控制协议**：改 patch 里 `strategy` 一行，重启（或 HMR）生效。线上参数映射见上表。

**实际效果**：`reasoningEffort: off` 时 Ollama 收到 `"think": false`，`low` 时收到 `"think": true`；Ollama 的思考轨迹以独立 reasoning 内容块回到 harness，正文与之分离。已用真实 dsh headless 链路实测验证。

## 配置参考

| 字段 | 默认 | 说明 |
|---|---|---|
| `providers` | `['ollama']` | 注册的提供方路由名（agent 的 `provider`） |
| `endpoint` | `http://127.0.0.1:11434` | Ollama 基地址 |
| `apiKeyEnv` | `''` | 承载 Bearer token 的**环境变量名**；空表示无凭据 |
| `strategy` | `native` | 思考控制策略，见映射表 |
| `nativeLevels` | `false` | `native` 下传 `think: "low"/"medium"/"high"` 而非 `true`（gpt-oss 系用） |
| `offSentinel` | `'none'` | `reasoning-effort` 下 `off` 的线上值；`'omit'` 表示不发字段 |
| `assumeThinking` | `auto` | 思考能力：`auto` 跟随 `/api/show` 的 `thinking`（不可达时视为有）；`yes`/`no` 强制 |
| `efforts` | `off/low/medium/high` | 公示的档位；条目为字符串或 `{id, name, description, budget}`，`budget` 供 `template-kwarg` 的 `thinking_budget` 使用 |
| `defaultEffort` | — | 请求未带档位时物化的默认档位；不配则保持提供方默认 |
| `defaultContextWindow` | `32768` | `/api/show` 无 `context_length` 时的回退上下文窗口 |
| `defaultMaxTokens` | `8192` | 请求未指定时的单次输出上限 |
| `streamIdleTimeoutMs` | `300000` | 流式读取的空闲超时（超时→`TIMEOUT` 错误） |
| `historyThinking` | `strip` | 助手历史中的 reasoning 块是否回传；`keep` 时走 Ollama 的 `message.thinking` 字段 |
| `imageCapability` | `auto` | 图片输入能力：`auto` 跟随 `/api/show` 的 `vision`；`yes`/`no` 强制 |
| `imageMaxPixels` | `1048576` | 单张请求图的像素预算（宽×高，等比投影） |
| `imageMaxBytes` | `4194304` | 单张请求图重编码后的字节目标 |
| `imageMaxPerRequest` | `8` | 单请求图片张数上限，超出从最旧卸载；`0` 关闭图片输入 |
| `imageMaxRequestBytes` | `33554432` | 单请求累计图片字节上限 |
| `models` | `[]` | 选型目录覆盖：`{id, name, description, contextWindow}`，也是 `/api/tags` 不可达时的回退目录 |

## 测试与验证

```sh
pnpm check   # tsc --noEmit
pnpm test    # node --test：26 个用例
```

测试通过脚本化 mock（`test/mock-ollama.mjs`，提供 `/api/tags`、`/api/show`、NDJSON `/api/chat`、SSE `/v1/chat/completions`）覆盖：四种策略的线上映射、`<think>` 标签分离、工具调用与 `role:tool` 历史、图片 base64/data-URI 载荷与预算卸载、`done_reason: length → max-tokens`、流中错误与 HTTP 400/500 → 稳定错误码、中止与空闲超时，以及**真实 `LlmRuntime`** 的档位校验与 `defaultEffort` 物化。

对真实 Ollama：把 `endpoint` 指回 `http://127.0.0.1:11434` 按上面安装即可。若要复现完整 dsh 链路又不想动本机配置，可用 mock + 隔离目录：

```sh
node test/mock-serve.mjs 21434 &
DSH_HOME=<隔离目录> dsh --profile <你的profile> "Reply with exactly: PONG"
```

## 已知边界

- 图片仅支持 **user 消息输入侧**；工具结果里的图片降级为文本占位，助手输出仍为文本（Ollama 视觉输入的形态决定）。
- `soft-switch` 无法表达思考强度分级：`off` 之外的档位在软开关层面等价于 `/think`；且 Qwen3-2507 分叉版不支持软开关，请用 `native`。
- 能力探测依赖 `/api/show`；Ollama Cloud 等不回 capabilities 的端点请用 `assumeThinking` / `imageCapability` 显式声明。

## 文件结构

```
src/index.ts      插件入口（TS 源）：Config schema + registerAdapter
src/adapter.ts    OllamaThinkAdapter（能力声明 / 图片预备 / stream 分发）
src/efforts.ts    统一档位词表与四种策略的线上映射
src/messages.ts   harness 消息 → 线上消息（system/user/assistant/tool、图片、reasoning）
src/native.ts     /api/chat 请求构建 + NDJSON 解析
src/openai.ts     /v1/chat/completions 请求构建 + SSE 解析
src/think-tag.ts  <think>…</think> 增量分离器
src/http.ts       归因头、凭据解析、空闲超时读行、错误码映射
src/config.ts     配置归一化
dist/             编译产物（pnpm build 生成，包安装的加载入口）
```
