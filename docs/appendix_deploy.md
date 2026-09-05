# 附录 B · 在线部署指南

本站点是纯静态的 docsify 应用（无需构建），有两种发布方式。

## 方式一：GitHub Pages（推荐，零构建）

1. 把整个仓库推到 GitHub（当前目录还不是 git 仓库，先 `git init` 并提交）
2. 仓库 Settings → Pages → Source 选 **Deploy from a branch**
3. Branch 选 `main`，目录选 **`/docs`**，保存
4. 一分钟后访问：

```text
https://<你的用户名>.github.io/<仓库名>/
```

发布范围是 `docs/` 整个目录，任何 md 增改推送后自动更新（Pages 缓存约 1~10 分钟）。

## 方式二：本地预览

```bash
# 仓库根目录执行
python -m http.server 3000
# 浏览器打开 http://localhost:3000/docs/
```

## 方式三：其他静态托管

`docs/` 目录直接拖给 Vercel / Netlify / Cloudflare Pages 即可，
无需任何构建命令，输出目录填 `docs`。

## 常见问题

- **站点 404**：确认 Pages 的目录选的是 `/docs` 而不是 `/ (root)`
- **图片/链接失效**：docsify 用 hash 路由（`#/chapter0/...`），
  站内链接请保持相对路径写法
- **CDN 依赖**：index.html 引用了 jsdelivr CDN（docsify 本体与插件），
  内网环境需自行下载这些脚本到 `docs/vendor/` 并改本地引用
