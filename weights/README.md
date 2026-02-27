# Weights And Large Artifacts

Большие файлы не хранятся в git.  
Ниже перечислены канонические архивы для воспроизведения и их статус публикации.

## Канонические архивы

Google Drive folder (all archives):
- https://drive.google.com/open?id=1Jgb1xTOhgvdicsL0oPyxplWl9yIP98-b

| Archive | Purpose | Local Source | Size | Public Link | SHA256 |
|---|---|---|---:|---|---|
| `team_bundle_top_new_tinder_plus_ckpts_2026-02-26_v1.zip` | полный data+checkpoint bundle | `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_bundle_top_new_tinder_plus_ckpts_2026-02-26_v1.zip` | 4.74 GB (5,090,949,660 bytes) | https://drive.google.com/open?id=1UM8CxgTj0iiTG9r9R7dWxFHjXV-ggHXh | `e596d98003a86cdc7f8385fc4d02847af3fcc0ef277dcb02bcb29566c556c584` |
| `team_final3_cv5_split_bundle_2026-02-26_v1.zip` | split-training bundle | `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/team_final3_cv5_split_bundle_2026-02-26_v1.zip` | 724.57 MB (759,768,207 bytes) | https://drive.google.com/open?id=1R0mseuvQKeb1kBDNv5WwG15Cvn55Lu9n | `d820064750e54abdfaa1a3b357ff1735698a2f9f93eebaccb96702c092fb2fec` |
| `convnext_small_deadline14f4_weights_f0f1_2026-02-26.zip` | lightweight weight-only bundle | `/Users/fedorgracev/Desktop/NeuralNetwork/dl_lab1/share/convnext_small_deadline14f4_weights_f0f1_2026-02-26.zip` | 755.31 MB (791,999,554 bytes) | https://drive.google.com/open?id=1q454m4zte12VFCeiBRo25RWSnKGTH2nl | `208575dd867f98b97915412d55f260c79d858259d66636c772b617910bedd0f5` |

Google Drive upload automation:
- `repro/scripts/upload_to_google_drive.py`
- usage example: `repro/README.md`

## Требования к публикации ссылок

- Доступ: "Anyone with the link / Доступ по ссылке".
- Желательно публиковать и URL, и SHA256.
- Держите этот файл в актуальном состоянии при обновлении артефактов.
