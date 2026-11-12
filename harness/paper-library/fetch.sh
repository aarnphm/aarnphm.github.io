#!/usr/bin/env bash
# Fetch paper PDFs and archived vLLM discussions into the thoughts library.
set -u
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
pdf_dir="$repo_root/content/thoughts/papers"
pr_dir="$repo_root/content/thoughts/inference/vllm-prs"
cache_dir="$repo_root/.quartz-cache/paper-library"
mkdir -p "$pr_dir" "$pdf_dir" "$cache_dir"
fail=()
for id in 2602.06036 2607.05147 2605.29707 2608.20530 2608.02438 2609.06498 2512.17351 2606.03825 2609.19969 \
          2512.24880 2601.07372 2601.18089 2603.15031 2409.19606v3 2608.30320; do
  out="$pdf_dir/$id.pdf"
  [ -s "$out" ] && { echo "have $id"; continue; }
  if curl -fsSL --max-time 120 -A "Mozilla/5.0" -o "$out.tmp" "https://arxiv.org/pdf/$id" && head -c4 "$out.tmp" | grep -q '%PDF'; then
    mv "$out.tmp" "$out"; echo "got $id"
  else rm -f "$out.tmp"; fail+=("arxiv:$id"); fi
done
while IFS= read -r n; do
  f="$pr_dir/$n.md"; [ -s "$f" ] && continue
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
done < "$script_dir/prs.txt"
printf '%s\n' "${fail[@]:-}" > "$cache_dir/FAILED.txt"
echo "done. failures: ${#fail[@]} (see $cache_dir/FAILED.txt)"
