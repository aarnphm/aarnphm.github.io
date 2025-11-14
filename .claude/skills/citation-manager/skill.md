# Citation Manager

Manage academic citations and the References.bib file for this digital garden.

## When to Use

- User requests to add a citation from arXiv, DOI, or raw BibTeX
- User wants to validate or format References.bib
- User needs to insert citation references in markdown files
- User wants to check if a paper is already cited in the bibliography

## Instructions

### Adding Citations from arXiv

1. Extract the arXiv ID from the URL or user input (e.g., "2301.00001")
2. Fetch BibTeX entry:
   ```bash
   curl https://arxiv.org/bibtex/<arxiv-id>
   ```
3. Read `content/References.bib` to check for duplicates
4. Append the new BibTeX entry to `content/References.bib`
5. Run `pnpm format` to organize and validate the bibliography
6. Confirm the citation key that can be used in markdown files

### Adding Citations from DOI

1. Extract the DOI from the user input
2. Use the citation-js library (already in dependencies) or curl:
   ```bash
   curl -LH "Accept: application/x-bibtex" https://doi.org/<doi>
   ```
3. Read `content/References.bib` to check for duplicates
4. Append the new BibTeX entry to `content/References.bib`
5. Run `pnpm format` to organize and validate
6. Confirm the citation key

### Adding Raw BibTeX

1. Validate the BibTeX format
2. Read `content/References.bib` to check for duplicate keys
3. Append the entry to `content/References.bib`
4. Run `pnpm format` to organize and validate
5. Confirm the citation key

### Inserting Citations in Markdown

1. Identify the citation key from References.bib
2. Use the syntax `[@key]` for in-text citations
3. Multiple citations: `[@key1; @key2]`
4. With page numbers: `[@key, p. 42]`

### Validating References.bib

1. Run `pnpm format` which includes bibtex-tidy
2. Check for errors in the output
3. Verify duplicate handling (by key, DOI, citation)

## Examples

### Example 1: Adding arXiv Paper

```
User: Add the paper arxiv:2301.00001 to my bibliography

1. Fetch: curl https://arxiv.org/bibtex/2301.00001
2. Read content/References.bib
3. Check for duplicates
4. Append entry
5. Run: pnpm format
6. Respond: "Added citation with key 'AuthorYear'. Use [@AuthorYear] to cite."
```

### Example 2: Citing in Markdown

```
User: How do I cite the attention paper in my notes?

1. Search content/References.bib for "attention" related papers
2. Find the citation key (e.g., "vaswani2017attention")
3. Respond: "Use [@vaswani2017attention] in your markdown file"
```

### Example 3: Validate Bibliography

```
User: Can you check my bibliography for errors?

1. Run: pnpm format
2. Check output for errors or warnings
3. Report results to user
```

## Notes

- The `pnpm format` command runs bibtex-tidy with specific options (see package.json)
- References.bib is processed with: sort by type/year/eprint, merge duplicates, no escaping
- Citation syntax follows Pandoc/rehype-citation conventions
- All academic papers should be stored in `content/thoughts/papers/<id>.pdf`
