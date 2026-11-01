# A_Memorix 同步说明

## 当前约定

- A_Memorix 主线源码位于 `src/A_memorix`
- 宿主接入层位于 `src/services/memory_service.py`、`src/webui/routers/memory.py` 与 dashboard 长期记忆页面
- 运行配置位于 `config/a_memorix.toml`
- 运行数据由 `storage.data_dir` 决定（当前配置模板默认 `data/a-memorix/`）
- 旧离线脚本默认目录仍可能落在 `data/plugins/a-dawn.a-memorix/`（见脚本注释/参数说明）
- Web 上传暂存目录为 `data/memory_upload_staging/`
- 上游同步方式固定为 `git subtree`

## 首次接入

```bash
./scripts/sync_a_memorix_subtree.sh add
```

默认同步源：

- 远端：`https://github.com/A-Dawn/A_memorix.git`
- 分支：`MaiBot_branch`
- 前缀：`src/A_memorix`

## 后续更新

```bash
./scripts/sync_a_memorix_subtree.sh pull
```

等价命令：

```bash
git subtree pull --prefix=src/A_memorix https://github.com/A-Dawn/A_memorix.git MaiBot_branch --squash
```

## 本地修改边界

- `src/A_memorix` 只保留上游源码和必须的宿主兼容补丁
- 宿主接入、配置暴露、图谱页与控制台页优先放在 MaiBot 侧 host service / memory router / dashboard 页面中
- 不再通过插件框架特判承载 A_Memorix

### 2026-11-01 临时兼容补丁

已核对上游 `MaiBot_branch` 的 `e54bf256dd984d174eaf5fc7b6a420db8b6a78a8`；
该版本仍未包含当前 `core/runtime/services` 和事实去重实现，不能直接用整包同步覆盖当前运行时。
按 `MODIFICATION_POLICY.md` 的本地例外，暂时保留以下最小接线修复，待上游目录结构对齐后移交：

- 人物事实去重与总结导入改用运行时 `_paragraph_store()`，遵循当前单池/双池模式。
- 图谱实体继续使用独立图谱池；不向已清空的兼容单池回填，不修改历史嵌入模型空间。
- MaiBot 集成测试覆盖单池、双池及人物事实中断恢复路径。

## 同步后检查

1. 确认 `config/a_memorix.toml` 中 `storage.data_dir` 与你的运行目录规划一致（默认模板为 `data/a-memorix`）
2. 运行 `python src/A_memorix/scripts/runtime_self_check.py --help`
3. 运行 `python -m pytest pytests/A_memorix_test/test_memory_service.py`
4. 运行 `cd dashboard && npm run test -- src/routes/resource/__tests__/knowledge-base.test.tsx`
