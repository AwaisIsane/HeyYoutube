# Yt AI companion

> 🧪 **Beta (v0.1.0)** — actively evolving. Expect rough edges.

A Chrome extension that turns any YouTube video into something you can **summarize,
quiz yourself on, and ask questions about** — all from a side panel next to the video.

It runs entirely **on-device** using **Gemini Nano**, the AI model built right into
Chrome. No servers, no API keys, no sign-up — your data never leaves your browser.

## Demo

[![Watch the demo](https://img.youtube.com/vi/Oeye29-mwzw/maxresdefault.jpg)](https://www.youtube.com/watch?v=Oeye29-mwzw)

▶️ **[Watch the demo on YouTube](https://www.youtube.com/watch?v=Oeye29-mwzw)**

## What it does

Open any YouTube video, click the extension icon, and a side panel appears with three modes:

- **📝 Summarize** — a clear, key-point summary of the whole video.
- **🎓 Review** — an auto-generated multiple-choice quiz to test what you learned.
- **💬 Query** — ask questions about the video and get answers, with multi-turn follow-ups.

The extension reads the video's transcript, so there's nothing to copy, paste, or upload.

## Why on-device?

Everything runs locally through Chrome's built-in **Gemini Nano** model:

- **Private** — the transcript and your questions stay on your machine.
- **Free** — no API keys, no usage limits, no accounts.
- **Offline-capable** — once the model is downloaded, it works without a network round-trip. would be fast if you have a beefy gpu

## Requirements

**Chrome 138+** with Gemini Nano available on-device. See Chrome's official setup guide
for hardware requirements :
[Built-in AI / Prompt API](https://developer.chrome.com/docs/ai/built-in).

## Install (from source)

```bash
pnpm install
pnpm fetch-model   # downloads embedding model + ONNX runtime into public/models
pnpm build         # type-check + bundle into dist/
```

Then load it unpacked:

1. Open `chrome://extensions`, enable **Developer mode**.
2. **Load unpacked** → select the `dist/` folder.
3. Open a YouTube video and click the extension icon to open the side panel.

---

*Built on Chrome's [built-in AI / Prompt API](https://developer.chrome.com/docs/ai/built-in) and Gemini Nano.*
