from quartz.embed_build import (
  chunk_document,
  count_tokens,
  format_document_texts,
  validate_token_limits,
)
import pytest


def test_chunks_respect_the_requested_token_budget():
  text = ' '.join(
    f'Passage {i} discusses gradient optimization in neural networks.'
    for i in range(80)
  )
  chunks = chunk_document(
    {'slug': 'probe', 'title': 'Optimization', 'text': text},
    chunk_size=128,
    overlap_tokens=32,
    model_id='google/embeddinggemma-300m',
    model_max_tokens=2048,
  )
  assert len(chunks) > 1
  assert all(0 < count_tokens(chunk['text']) <= 128 for chunk in chunks)


def test_chunking_preserves_short_sections_and_the_end_of_a_note():
  text = (
    'alpha beta gamma delta\n\n' * 8
  ) + 'The unique final fact is zephyr.'
  chunks = chunk_document(
    {'slug': 'short-sections', 'title': 'Sections', 'text': text},
    chunk_size=12,
    overlap_tokens=2,
  )
  assert chunks
  assert any('zephyr' in chunk['text'] for chunk in chunks)
  assert all(count_tokens(chunk['text']) <= 12 for chunk in chunks)


def test_document_prompts_include_titles_for_both_gemma_generations():
  for model in ('google/embeddinggemma-300m', 'google/embeddinggemma-2'):
    assert format_document_texts(['a passage'], model, ['Attention']) == [
      'title: Attention | text: a passage'
    ]
    assert format_document_texts(['a passage'], model) == [
      'title: none | text: a passage'
    ]


def test_model_budget_includes_the_actual_document_title():
  with pytest.raises(ValueError, match='token limit'):
    validate_token_limits(
      ['short passage'], 32, 'google/embeddinggemma-300m', ['title ' * 64]
    )


def test_chunking_rejects_an_impossible_overlap():
  with pytest.raises(ValueError, match='overlap'):
    chunk_document(
      {'slug': 'bad-config', 'text': 'a passage ' * 100},
      chunk_size=16,
      overlap_tokens=16,
    )
