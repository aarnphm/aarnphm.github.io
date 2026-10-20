# image-to-react-port

Continuation of the Opus figure ports on `claude/loving-wozniak-s7v18s`. Adds six figures for remaining note embeds, deletes unused diagram assets from the Opus ports, and records harness HTML plus Helium screenshots.

## Replay

```bash
node docs/verification/image-to-react-port/render.mjs
```

Serve the repo root and open each `artifacts/<Component>/page.html`. Screenshots in this folder were taken from those pages in Helium.

## Components

| Note | Component | Replaced image |
| --- | --- | --- |
| `content/research.md` | `FlashInferPageLayout` | `page_layout_flashinfer.webp` |
| `content/research.md` | `VllmEngineLayout` | `request-life-in-vllm.webp` |
| `thoughts/Wireshark.md` | `PacketSniffer` | `packet-sniffer.webp` |
| `thoughts/Transformers.md` | `MTPModules` | `MTP-deepseek.webp` |
| `thoughts/Attribution parameter decomposition.md` | `ParamDecomposition` | `apd.webp` |
| `thoughts/Alignment.md` | `SafetyLevels` | `alignment-asl-scale.webp` |
