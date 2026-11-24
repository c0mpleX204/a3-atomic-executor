# A3 六场景评测数据

本目录包含 2026 年 11 月 24 日完成的 186 个回合。对应的场景结果和单回合 API 统计见[评测报告](../../reports/a3-six-scenario-evaluation-2026-11-24.md)。

[下载原始评测数据 ZIP](raw-results.zip)。解压后按模式、场景和 seed 查阅。

- `manifest.json` 记录评测设置和计划回合。
- `summary.json` 汇总结果，其 `report` 字段可直接定位到压缩包内。
- `export-index.json` 列出每个回合及其文件。
- `episodes/` 包含 186 份回合报告、186 份 API 用量记录、183 份事件日志和 253 个状态检查点。3 个启动失败回合在报告中保留。

发布版清理了实例目录，并把报告引用改为包内路径。部署日志、录像和截图保存在本地归档。

ZIP 的 SHA-256 为 `051526f0ee53ea4a22eb51578f929c62fb9a69edd3697332f911aca1db66b045`。
