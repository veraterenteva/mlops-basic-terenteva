# HW2: Git Branches (Gitflow)

Домашнее задание №2 — работа с ветками и Pull Request по упрощённому
[Gitflow](https://nvie.com/posts/a-successful-git-branching-model/).

## Схема ветвления

| Ветка | Назначение | Откуда создаётся | Куда мержится |
| --- | --- | --- | --- |
| `main` | стабильные релизы, защищена | — | — |
| `develop` | интеграция изменений | `main` | `main` (через `release/*`) |
| `feature/<name>` | новая функциональность | `develop` | `develop` (через PR) |
| `release/<version>` | подготовка релиза | `develop` | `main` и `develop` |
| `hotfix/<name>` | срочное исправление | `main` | `main` и `develop` |

## Сквозной сценарий (команды)

```bash
# 1. Базовая ветка разработки
git switch main
git switch -c develop

# 2. Фича: добавляем функцию mean и тесты
git switch -c feature/add-mean develop
#   ... правки в hw1/src/mlops_hw1 и hw1/tests ...
git add .
git commit -m "feat: add mean function with tests"
git push -u origin feature/add-mean
#   -> Pull Request feature/add-mean -> develop

# 3. Ещё одна фича: проверка типов
git switch -c feature/typing develop
#   ... правки ...
git commit -m "chore: add type annotations"
git push -u origin feature/typing
#   -> Pull Request feature/typing -> develop

# 4. Релиз
git switch -c release/0.2.0 develop
#   ... обновляем версию в pyproject.toml ...
git commit -m "chore(release): 0.2.0"
git switch main
git merge --no-ff release/0.2.0
git tag -a v0.2.0 -m "v0.2.0"
git switch develop
git merge --no-ff release/0.2.0
git branch -d release/0.2.0

# 5. Hotfix после релиза
git switch -c hotfix/0.2.1 main
#   ... исправление ...
git commit -m "fix: handle empty input"
git switch main
git merge --no-ff hotfix/0.2.1
git tag -a v0.2.1 -m "v0.2.1"
git switch develop
git merge --no-ff hotfix/0.2.1
git branch -d hotfix/0.2.1
```

## Соглашение о коммитах

Используется [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`.

## Проверка результата

```bash
git log --graph --oneline --decorate --all
git branch -a
git tag -n
```
