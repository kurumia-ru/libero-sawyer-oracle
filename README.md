# openpi-sawyer-oracle-code

LIBERO Object（Task0 / Task1）を Sawyer で通すために使ったコードです。  
scripted デモ作成、BC 学習まわりのランチャ、DAgger、評価スクリプトが入っています。

https://github.com/kurumia-ru/libero-sawyer-oracle

## 結果の目安

| 段階 | 内容 | Task0 / Task1 |
|------|------|----------------|
| Scripted BC (`osc_v8`) | open-loop 成功デモ + 相対 OSC | 約 0% / 4% |
| + DAgger closefail | 掴み手前の失敗を expert で直して再学習 | 約 28% / 66% |

評価の開始姿勢は学習デモに合わせて `home_w30` です。  
π0.5 の学習本体は別リポジトリ（openpi）側を使います。

## 主なファイル

| ファイル | 役割 |
|----------|------|
| `export_scripted_demos.py` | scripted デモ作成 |
| `collect_dagger_takeover.py` | 失敗箇所から expert が引き継いで軌道収集 |
| `eval_libero_sawyer.py` | Sawyer 評価 |
| `libero_grasp_smoke.py` | 掴み・配置モーション |
| `scripted_action_spaces.py` | delta OSC / abs EEF |
| `sawyer_ik.py` | Sawyer IK |
| `merge_lerobot_repos.py` | DAgger データの結合 |
| `assets/robots/sawyer/` | Sawyer の MuJoCo 資産 |
| `run_*.sh` | 実験時のパイプライン |

## 依存

- LIBERO + robosuite / MuJoCo
- LeRobot
- openpi（π0.5 学習・`serve_policy`）と `openpi_client`
- 任意: PyRoki（`install_pyroki_for_ik.sh`）

## パス

デフォルトは次の置き場を想定しています。環境に合わせて変数で上書きしてください。

| 変数 | デフォルト | 意味 |
|------|------------|------|
| `OPENPI_EVAL_ROOT` | `~/openpi_ws/openpi_eval` | このリポジトリ |
| `OPENPI_LIBERO_ROOT` | `~/openpi_ws/openpi/third_party/libero` | LIBERO |
| `LIBERO_DATA_ROOT` | `~/openpi_data` | データセット |
| `OPENPI_XEMB_ROOT` | `~/openpi_ws/openpi-xemb` | 学習用 openpi |

`run_*.sh` 内の `ROOT` / `DATA` も同じ前提です。

## 動かし方（要約）

1. デモ: `run_export_scripted_object_task.sh`（`TASK_ID=0` または `1`）
2. BC: `run_osc_v8_obj0_pipeline.sh` など
3. 評価: `eval_libero_sawyer.py`
4. DAgger: `run_dagger_obj0_closefail_pipeline.sh` → merge → 再学習  
   Task1 は `run_dagger_r2_closefail_pipeline.sh`

## ライセンス

実験再現用に公開しています。Sawyer メッシュなどは上流（robosuite / Rethink 系）のライセンスに従ってください。π0.5 / openpi は各上流のライセンスに従います。
