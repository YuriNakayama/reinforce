---
applyTo: "**"
---

# コミットメッセージの書き方

- **日本語で対応する**: コミットメッセージは日本語で記述します。
- **変更内容を簡潔に説明する**: 何を変更したのか、なぜその変更が必要だったのかを明確にします。
- **一貫性を保つ**: プロジェクト全体でコミットメッセージのスタイルを統一します。以下のプレフィックスを使用してください。

|         Emoji         | code                    | Mean                                    | File Example                 |
| :-------------------: | :---------------------- | :-------------------------------------- | ---------------------------- |
|      :sparkles:       | `:sparkles:`            | Introduce new features.                 | all files                    |
|         :zap:         | `:zap:`                 | Improve performance                     | all files                    |
|         :art:         | `:art:`                 | Improve structure / format of the code. | formatted by prettier / ruff |
|       :recycle:       | `:recycle:`             | Refactor code.                          | all files                    |
|  :white_check_mark:   | `:white_check_mark:`    | Add, update, or pass tests.             | test files                   |
|         :bug:         | `:bug:`                 | Fix a bug.                              | bug fix                      |
|      :ambulance:      | `:ambulance:`           | Critical hotfix                         | bug fix                      |
|        :books:        | `:books:`               | Add or update documentation.            | docs / .md                   |
| :construction_worker: | `:construction_worker:` | Add or update CI build system.          | workflows                    |
|       :bricks:        | `:bricks:`              | Infrastructure related changes          | infrastructure as code files |
|       :wrench:        | `:wrench:`              | Add or update configuration files.      | config files                 |
|   :heavy_plus_sign:   | `:heavy_plus_sign:`     | Add a dependency.                       | dependency addition          |
