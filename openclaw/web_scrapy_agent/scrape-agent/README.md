# scrape-agent Skill 安装说明（Windows）

## 目录结构

安装完成后，你的文件应该是这样的：

```
C:\Users\你的用户名\
├── .env                          ← 放 OPENAI_API_KEY
└── openclaw-skills\
    └── scrape-agent\
        ├── SKILL.md              ← OpenClaw 读取的技能描述
        └── scrape_agent.py       ← 你的爬虫脚本
```

---

## 第一步：安装 Python 依赖

打开 PowerShell 或命令提示符，运行：

```bash
pip install openai requests beautifulsoup4 python-dotenv
```

---

## 第二步：创建 .env 文件

在 `C:\Users\你的用户名\` 下创建 `.env` 文件，内容：

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

---

## 第三步：放置文件

把 `SKILL.md` 和 `scrape_agent.py` 放到：
```
C:\Users\你的用户名\openclaw-skills\scrape-agent\
```

---

## 第四步：测试脚本是否正常工作

在 PowerShell 中运行：

```bash
python "$env:USERPROFILE\openclaw-skills\scrape-agent\scrape_agent.py" "https://example.com"
```

如果输出了网页摘要，说明脚本正常。

---

## 第五步：注册 Skill 到 OpenClaw

在终端中运行（OpenClaw 需已安装）：

```bash
openclaw skill add "%USERPROFILE%\openclaw-skills\scrape-agent"
```

或者在 OpenClaw 的 `openclaw.json` 中手动添加：

```json
{
  "skills": {
    "paths": [
      "%USERPROFILE%\\openclaw-skills\\scrape-agent"
    ]
  }
}
```

---

## 第六步：通过 WhatsApp 测试

发送任意 URL 给你的 WhatsApp Bot，例如：
```
https://www.bbc.com/news
```

OpenClaw 会自动识别 URL，调用 Skill，返回 GPT 摘要给你。

---

## 常见问题

**Q: 找不到 python 命令？**
确保 Python 已加入 PATH，或用 `py` 替换 `python`，并修改 SKILL.md 中的命令。

**Q: OpenAI API key 报错？**
检查 `C:\Users\你的用户名\.env` 文件内容是否正确，没有多余空格。
