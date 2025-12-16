# MDAgent Website

这是 MDAgent 项目的 GitHub Pages 网站。

## 部署说明

1. 在 GitHub 仓库的 Settings > Pages 中：
   - Source 选择 "GitHub Actions"
   - 保存设置

2. 当推送代码到 main 分支时，GitHub Actions 会自动构建并部署网站

3. 网站将发布到：`https://[你的用户名].github.io/PKU_MDAgent/`

## 本地预览

如果你想在本地预览网站，可以使用 Python 的简单 HTTP 服务器：

```bash
cd docs
python -m http.server 8000
```

然后在浏览器中访问 `http://localhost:8000`

