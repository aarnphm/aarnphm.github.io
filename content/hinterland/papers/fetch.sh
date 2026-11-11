#!/usr/bin/env bash
# run from this folder (hinterland/papers). PDFs land here, as hinterland/papers/<arxiv-id>.pdf.
set -u
cd "$(dirname "$0")"
mkdir -p prs blogs
fail=()
for id in 2602.06036 2607.05147 2605.29707 2608.20530 2608.02438 2609.06498 2512.17351 2606.03825 2609.19969 \
          2512.24880 2601.07372 2601.18089 2603.15031 2409.19606v3 2608.30320; do
  out="$id.pdf"
  [ -s "$out" ] && { echo "have $id"; continue; }
  if curl -fsSL --max-time 120 -A "Mozilla/5.0" -o "$out.tmp" "https://arxiv.org/pdf/$id" && head -c4 "$out.tmp" | grep -q '%PDF'; then
    mv "$out.tmp" "$out"; echo "got $id"
  else rm -f "$out.tmp"; fail+=("arxiv:$id"); fi
done
while read -r name url; do
  curl -fsSL --max-time 60 -A "Mozilla/5.0" -o "blogs/$name.html" "$url" || fail+=("blog:$url")
done <<'B'
dflash2-inco https://inco.ai/blog/dflash2/
replayssm-tridao https://tridao.me/blog/2026/replayssm/
replayssm-fork-readme https://github.com/Johnny-Liou/ReplaySSM
qwen38-flash-next-swift https://swift.readthedocs.io/en/latest/BestPractices/Qwen3_8-Flash-Next-Best-Practice.html
qwen38-flash-fonearena https://www.fonearena.com/blog/490674/qwen3-8-flash-features.html
flashinfer-checkpointing-ssu https://docs.flashinfer.ai/generated/flashinfer.mamba.checkpointing_ssu.html
B
for n in $(cat prs.txt); do
  f="prs/$n.md"; [ -s "$f" ] && continue
  if gh pr view "$n" -R vllm-project/vllm --json number,title,author,state,createdAt,mergedAt,url,body,comments \
       --template '# #{{.number}} {{.title}}

{{.url}} · {{.author.login}} · {{.state}} · created {{.createdAt}} · merged {{.mergedAt}}

{{.body}}

---
## comments
{{range .comments}}
**{{.author.login}}** ({{.createdAt}}):

{{.body}}

{{end}}' > "$f" 2>/dev/null; then :;
  elif gh issue view "$n" -R vllm-project/vllm --json number,title,author,state,createdAt,url,body,comments \
       --template '# #{{.number}} {{.title}}

{{.url}} · {{.author.login}} · {{.state}} · created {{.createdAt}}

{{.body}}

---
## comments
{{range .comments}}
**{{.author.login}}** ({{.createdAt}}):

{{.body}}

{{end}}' > "$f" 2>/dev/null; then :;
  else rm -f "$f"; fail+=("gh:#$n"); fi
done
printf '%s\n' "${fail[@]:-}" > FAILED.txt
echo "done. failures: ${#fail[@]} (see FAILED.txt)"
