# indexing exclusions, 2026-10-26

base: `143f9f5f`, branch: `fix/indexing-discovery`. deployment validation is pending.

## observed

- the october 25 search console email lists robots.txt, soft 404, and other 4xx exclusions, but contains no example urls. the available browser shows the signed-out search console landing page. these report categories remain unattributed.
- deployed robots.txt returns 200 on the apex, notes, stream, and t hosts. the wildcard group allows `/`; the two perplexity exclusions remain intentional and unchanged.
- deployed sitemap.xml returns 200 and parses as xml with 1,945 urls. it includes `/security-policy` and `/posts/iron-ring`, whose live html returns 200 with `noindex, nofollow`. source comparison also finds other restricted entries. the sitemap contradiction is confirmed, but is not evidence that these urls caused the three reported categories.
- `/404` returns 200, an error page titled `introuvable`, no canonical, and no noindex. a deliberately nonexistent path returns a genuine 404. this establishes a direct-error-page defect rather than a general missing-route fallback defect.
- `/are.na` returns 301 to `/arena`, then 200 with the arena canonical. `/thoughts` returns 307 to `/thoughts/`, then 200 with its canonical. `/dump` returns 404; no redirect was restored without confirming the currently intended mapping.
- `/uses`, the homepage, and 12 systematically spaced sitemap entries return 200. all 12 sampled sitemap entries have matching canonicals and no noindex. this is a sample, not a whole-site crawl or proof of google indexing.
- current deployment is cloudflare workers static assets, configured with `not_found_handling = '404-page'`. terminal delivery preserves origin status codes. quartz aliases generate server-side 308 rules alongside html fallback redirects. spa navigation fetches html and changes browser history; it cannot set the initial document's http status.
- python urllib requests returned uniform 17-byte 403 responses, including for resources that curl fetched successfully. these client-path responses were excluded from site diagnosis; they do not establish googlebot blocking. no verified googlebot or search console live test was available.

## changes

1. filter noindex and protected frontmatter at sitemap generation, preserving the shared link/search index and public generated entries. safely handle generated entries without fileData or dates.
2. mark the generated error page noindex.
3. convert successful direct `/404` and `/404.html` asset responses to 404 while retaining their bodies and headers. retain existing error and redirect responses.

robots policy, content restrictions, access controls, and public-page canonicals are unchanged.

## validation performed

the new sitemap regression test was written before the implementation. it exercises the real writer against a temporary output directory and checks public, explicitly public, noindex, protected, and metadata-free generated entries. it failed before the fix and passed afterward.

```sh
node --import tsx --test quartz/plugins/emitters/contentIndex/discovery.test.ts
node node_modules/oxfmt/bin/oxfmt --check quartz/plugins/emitters/contentIndex/discovery.ts quartz/plugins/emitters/contentIndex/discovery.test.ts quartz/plugins/emitters/404.tsx worker/index/terminal-delivery.ts
node node_modules/oxlint/bin/oxlint quartz/plugins/emitters/contentIndex/discovery.ts quartz/plugins/emitters/contentIndex/discovery.test.ts quartz/plugins/emitters/404.tsx worker/index/terminal-delivery.ts
git diff --check
```

all four checks pass. a separate minimal npm toolchain supplied the focused test dependencies. full repository installation was rejected by pnpm's supply-chain check because `pyright-root@0.0.0` lacks tarball integrity. the policy and lockfile were not changed. existing contentIndex and alias tests could not load `vfile` under the minimal toolchain. full types, suite, rendered error page, and worker-runtime checks are not claimed. no running local quartz watcher was available, and no full build or deployment was launched.

## verify before deployment

in an established development environment with its verified dependencies:

```sh
pnpm exec tsx --test quartz/plugins/emitters/contentIndex/discovery.test.ts quartz/plugins/emitters/contentIndex.test.ts quartz/plugins/emitters/aliases.test.ts
pnpm check
```

use the existing watcher and wait for the matching `build:ready`. test the cloudflare worker preview, not only a generic static file server. set the preview origin below:

```sh
origin=http://localhost:8080
curl -sS -D - "$origin/404" -o /tmp/indexing-404.html
curl -sS -L -D - "$origin/404.html" -o /tmp/indexing-404-html.html
curl -sS -D - "$origin/__indexing_probe_20261026_missing" -o /tmp/indexing-missing.html
curl -sS -D - "$origin/uses" -o /tmp/indexing-public.html
curl -sS "$origin/sitemap.xml" -o /tmp/indexing-sitemap.xml
```

acceptance: both explicit error paths finish at 404 and contain noindex; the nonexistent path stays 404; uses stays 200 with its canonical and no noindex. the sitemap must parse, omit restricted urls including security-policy and posts/iron-ring, and retain public entries. inspect the error page rendering and exercise a public internal navigation and a missing-page navigation. repeat these checks against production after an approved release, then inspect representative affected urls in search console before requesting validation.

## remaining evidence

obtain example urls plus last crawl dates from each of the three search console categories. for each, compare url inspection's fetched status, robots verdict, canonical, and rendered content with the current deployment. investigate cloudflare security events only if actual affected urls and crawl evidence show a 4xx denial. a current permissive robots file cannot establish what google saw at an earlier crawl or on an untested hostname.

google's guidance: https://developers.google.com/search/docs/crawling-indexing/troubleshoot-crawling-errors and https://developers.google.com/search/docs/crawling-indexing/javascript/fix-search-javascript.

live observations above were captured on october 26. repeat the curl checks to obtain fresh evidence; local raw captures are not included in this pull request.
