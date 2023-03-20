---
title: How to build this web
categories:
- Technology
tags:
- Hexo
- Icarus
---

## Install Node.js

- for linux [reference this](https://github.com/nodesource/distributions)
- for windows through [nvs](https://github.com/jasongin/nvs/) or [nvm](https://github.com/nvm-sh/nvm)
- for mac through [Homebrew](https://brew.sh/) or [MacPorts](http://www.macports.org/)

## Install Hexo (require Node.js)

```bash
npm install -g hexo-cli
```

## Create your site
```bash
hexo init <folder> 
cd <folder>
npm install
npm install -S hexo-theme-icarus hexo-renderer-inferno # install icarus
hexo config theme icarus # use theme icarus
hexo server # start server at localhost
```

## Reference

- [hexo-tutorial](https://hexo.io/zh-cn/docs/)
- [getting-started-with-icarus](https://ppoffice.github.io/hexo-theme-icarus/uncategorized/getting-started-with-icarus/)



