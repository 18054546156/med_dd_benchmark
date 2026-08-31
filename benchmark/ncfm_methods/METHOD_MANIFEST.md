# 方法文件清单

| 方法 | 主要实现文件 | 配置/启动位置 |
|---|---|---|
| 原始 NCFM | `ncfm_release/NCFM/NCFM.py`、`ncfm_release/condenser/` | `ncfm_release/config/`、`ncfm_release/condense/` |
| M16 | `m16_feature_map_token/NCFM/NCFM.py`、`m16_feature_map_token/condenser/` | `m16_feature_map_token/config/`、`m16_feature_map_token/scripts/` |
| M22 | `m22_token_attention/NCFM/NCFM.py`、`m22_token_attention/condenser/` | `m22_token_attention/config/`、`m22_token_attention/scripts/` |
| DR-LTM | `dr_ltm/NCFM/NCFM.py`、`dr_ltm/condenser/` | `dr_ltm/config/`、`dr_ltm/scripts/` |

每个目录是独立副本，不通过 Python import 依赖其他方法。数据、checkpoint 和运行产物不在 Git 包内。
