# HW2: Git Branching (Gitflow)

Домашнее задание №2 — работа с ветками и Pull Request по
[Gitflow](https://nvie.com/posts/a-successful-git-branching-model/).

## Что требовалось

1. Создать постоянную ветку разработки `develop` от `main`.
2. Завести минимум две ветки функциональности `feature/*`, отведённые от
   `develop`, и влить их обратно **только через Pull Request**.
3. Сделать релизную ветку `release/*`, влить её в `main` и `develop`,
   поставить аннотированный тег `vX.Y.Z`.
4. Сделать ветку `hotfix/*` от `main`, влить в `main` и `develop`,
   поставить новый тег.
5. Зафиксировать фактическое состояние в документации.

Соглашение о коммитах — [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`.

## Как это устроено сейчас

### Схема ветвления

| Ветка | Назначение | Откуда создаётся | Куда мержится |
| --- | --- | --- | --- |
| `main` | стабильные релизы | — | — |
| `develop` | интеграция изменений | `main` | `main` через `release/*` |
| `feature/<name>` | новая функциональность | `develop` | `develop` через PR |
| `release/<version>` | подготовка релиза | `develop` | `main` и `develop` |
| `hotfix/<name>` | срочное исправление | `main` | `main` и `develop` |

### Фактическое состояние веток

Все ветки отведены от `main` на коммите `81e9846`.

| Ветка | SHA | Коммит |
| --- | --- | --- |
| `main` | `81e9846` | оставлен один uv-проект в hw1 |
| `develop` | `0ddc16b` | docs: link created pull requests |
| `feature/add-rms` | `f56886e` | feat: add rms function |
| `feature/add-mode` | `5247218` | feat: add mode function |
| `release/0.2.0` | `5920942` | Merge branch 'develop' into release/0.2.0 |
| `hotfix/0.2.1` | `4c977e1` | fix: handle empty input |

История `develop` состоит из трёх коммитов:

```text
0ddc16b  docs: link created pull requests
c5c33ab  docs: describe gitflow plan and branch state
e17f6b0  chore: init develop branch for gitflow
```

История `release/0.2.0` — собственный коммит плюс merge из `develop`,
чтобы документация попала в релиз:

```text
5920942  Merge branch 'develop' into release/0.2.0
ac03172  chore(release): bump version to 0.2.0
```

### Открытые Pull Request'ы

| PR | Из | В | Описание |
| --- | --- | --- | --- |
| [#1](https://github.com/veraterenteva/mlops-basic-terenteva/pull/1) | `feature/add-rms` | `develop` | Ветка функциональности по Gitflow |
| [#2](https://github.com/veraterenteva/mlops-basic-terenteva/pull/2) | `feature/add-mode` | `develop` | Ветка функциональности по Gitflow |
| [#3](https://github.com/veraterenteva/mlops-basic-terenteva/pull/3) | `release/0.2.0` | `main` | Подготовка релиза |
| [#4](https://github.com/veraterenteva/mlops-basic-terenteva/pull/4) | `hotfix/0.2.1` | `main` | Срочное исправление |

PR созданы через GitHub REST API, потому что GitHub CLI в системе не установлен.

### Граф репозитория

```text
*   5920942 (origin/release/0.2.0) Merge branch 'develop' into release/0.2.0
|\
| * 0ddc16b (origin/develop) docs: link created pull requests
| * c5c33ab docs: describe gitflow plan and branch state
| * e17f6b0 chore: init develop branch for gitflow
* | ac03172 chore(release): bump version to 0.2.0
|/
| * 5247218 (origin/feature/add-mode) feat: add mode function
|/
| * 4c977e1 (origin/hotfix/0.2.1) fix: handle empty input
|/
| * f56886e (origin/feature/add-rms) feat: add rms function
|/
* 81e9846 (origin/main) оставлен один uv-проект в hw1
```

### Что сделано, а что нет

Сделано:

- созданы и запушены шесть веток;
- в каждой ветке есть коммит;
- `develop` обогащён документацией и влит в `release/0.2.0`;
- открыты четыре Pull Request'а;
- состояние описано в корневом README.

## Полный сценарий с командами

```bash
# 1. Базовая ветка разработки
git switch main
git switch -c develop

# 2. Фича
git switch -c feature/add-rms develop
#   ... правки ...
git commit -m "feat: add rms function"
git push -u origin feature/add-rms
#   -> Pull Request feature/add-rms -> develop

# 3. Ещё одна фича
git switch -c feature/add-mode develop
#   ... правки ...
git commit -m "feat: add mode function"
git push -u origin feature/add-mode
#   -> Pull Request feature/add-mode -> develop

# 4. Релиз
git switch -c release/0.2.0 develop
git commit -m "chore(release): bump version to 0.2.0"
git merge --no-ff develop
git switch main
git merge --no-ff release/0.2.0
git tag -a v0.2.0 -m "v0.2.0"
git switch develop
git merge --no-ff release/0.2.0
git branch -d release/0.2.0

# 5. Hotfix после релиза
git switch -c hotfix/0.2.1 main
git commit -m "fix: handle empty input"
git switch main
git merge --no-ff hotfix/0.2.1
git tag -a v0.2.1 -m "v0.2.1"
git switch develop
git merge --no-ff hotfix/0.2.1
git branch -d hotfix/0.2.1
```

## Как проверить

```bash
git log --graph --oneline --decorate --all
git branch -vv
git tag -n
git ls-remote --heads origin
```
