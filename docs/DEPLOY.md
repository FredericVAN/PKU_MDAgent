# GitHub Pages 部署说明

## 方法一：使用 GitHub Actions（推荐）

1. 在 GitHub 仓库中：
   - 进入 **Settings** > **Pages**
   - **Source** 选择 **"GitHub Actions"**
   - 保存设置

2. 推送代码后，GitHub Actions 会自动部署

3. 网站地址：`https://[你的用户名].github.io/PKU_MDAgent/`

## 方法二：直接从 docs 文件夹部署（更简单）

1. 在 GitHub 仓库中：
   - 进入 **Settings** > **Pages**
   - **Source** 选择 **"Deploy from a branch"**
   - **Branch** 选择 **"main"**
   - **Folder** 选择 **"/docs"**
   - 保存设置

2. 网站会自动从 docs 文件夹部署

3. 网站地址：`https://[你的用户名].github.io/PKU_MDAgent/`

## 注意事项

- 如果使用**方法二**，可以删除 `.github/workflows/deploy.yml` 文件（不需要 GitHub Actions）
- 确保 `docs/index.html` 文件存在
- 图片路径使用相对路径：`assets/xxx.png`

