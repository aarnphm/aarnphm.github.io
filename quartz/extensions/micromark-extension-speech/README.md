# French speech markup

Wrap a word, phrase, or sentence in `{{...}}` to make its text playable with a faint orange dotted underline:

```markdown
Pour vous présenter : {{Je m'appelle Aaron.}}

Compare {{tu}} et {{tout}}.

Existing inline formatting is retained: {{`Je viens du Canada.`}}
```

The syntax is available in Markdown notes and flashcard faces. Keep each marked phrase on one
source line. The enclosed Markdown remains selectable and readable when JavaScript or the speech
model is unavailable. Playback buttons start disabled and the browser runtime enables them.

## Parsing boundaries

This extension adds a micromark inline construct. Code blocks, code spans containing the delimiters,
LaTeX regions, escaped opening braces, empty spans, and unclosed spans keep their original meaning.
Inline code and emphasis _inside_ a marked phrase retain their formatting. Nested speech spans are
unsupported. The `sidenotes` prefix stays reserved for the existing sidenote syntax.

The rendered span has class `speech-phrase`, `lang="fr"`, and a `data-speech-text` attribute containing
the rendered words without Markdown punctuation. Its native `button.speech-play` contains those
words and has an accessible name of `Écouter : <phrase>`.

The text itself is a native button; there is no adjacent speaker icon. Its inline formatting remains intact.
Controls are omitted inside or around existing interactive elements and from cloze phrases containing a
hidden deletion. An annotation never exposes the answer on a flashcard front. Flashcard identity
ignores only the delimiters of syntactically valid speech spans, preserving the exact enclosed
Markdown in the content hash. Existing IDs and cloze sibling group IDs survive annotation-only
edits. Code examples that need speech should retain their existing code delimiter, for example
change `` `bonjour` `` to ``{{`bonjour`}}``.

LLM Markdown exports contain the enclosed text and formatting without playback markup.

## Hosting the model

The browser runtime owns model selection, caching, and playback. `lang="fr"` identifies the text's
language; it does not claim that a voice has a verified Canadian French accent.

Set a model mirror in `quartz.config.ts` without editing the notes:

```typescript
Plugin.Speech({ modelBaseUrl: '/static/speech/supertonic-2' })
```

The transformer puts this URL in `data-speech-model-base-url` for the runtime. The runtime's model
manifest defines the files that a mirror must provide.

## Verification

The parser check exercises the actual Quartz Markdown processor, Markdown-to-HTML bridge, and
HTML processor. It writes fixture inputs, rendered HTML, LLM output, and a JSON report. A captured
pre-annotation deck baseline verifies every existing card and sibling group ID and confirms that
only speech delimiters changed in the source.

```sh
pnpm exec tsx quartz/scripts/verify-french-speech.ts \
  --baseline quartz/.quartz-cache/speech-evidence/deck-baseline.json \
  --output quartz/.quartz-cache/speech-evidence
```

The baseline is an evidence artifact captured before corpus annotation, containing each deck's
path, original source, parsed card identities, and parse errors. Keep that artifact when repeating
the identity check. Browser playback and model caching need separate served-browser verification.
