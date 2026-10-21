"""Typed durable records and strict JSON boundaries for the coordinator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import json
import math
import types
from typing import Literal, TypeVar, get_args, get_origin, get_type_hints

Provider = Literal['sol', 'astra', 'opus', 'argon', 'caw']
Stage = Literal['trace', 'verify']
Status = Literal[
  'queued',
  'awaiting_review',
  'revise',
  'accepted',
  'blocked_source',
  'needs_attention',
  'stale',
]
T = TypeVar('T')


@dataclass
class Pin:
  file: str
  sha256: str


@dataclass
class Source:
  url: str
  sha256: str
  evidence: str


@dataclass
class Backend:
  command: str
  model: str | None
  effort: str | None


@dataclass
class Backends:
  sol: Backend
  astra: Backend
  opus: Backend
  argon: Backend
  caw: Backend

  def get(self, provider: Provider) -> Backend:
    return {
      'sol': self.sol,
      'astra': self.astra,
      'opus': self.opus,
      'argon': self.argon,
      'caw': self.caw,
    }[provider]


@dataclass
class TraceModel:
  provider: Literal['sol', 'argon', 'caw']
  model: str
  effort: str | None


@dataclass
class Limits:
  invocationMs: int
  workerMs: int
  maxCalls: int
  maxAttempts: int
  maxOutputBytes: int
  maxSvgBytes: int
  maxSegments: int
  maxDepth: int
  maxCoordinate: int
  maxRasterMs: int


@dataclass
class Tier:
  width: int
  height: int
  quality: int | None


@dataclass
class Exports:
  encoding: Literal['lossless-webp', 'lossy-webp']
  shelf: Tier
  detail: Tier


@dataclass
class Config:
  version: Literal[4]
  trace: TraceModel
  reviewers: list[Provider]
  providers: Backends
  limits: Limits
  exports: Exports

  def validate(self) -> None:
    if not self.reviewers or len(set(self.reviewers)) != len(self.reviewers):
      raise ValueError('Require distinct reviewers')
    additional = 'sol' if self.trace.provider == 'caw' else 'caw'
    if not {'astra', 'opus', additional}.issubset(self.reviewers):
      raise ValueError(f'Astra, Opus and {additional} are required reviewers')
    if self.trace.provider in self.reviewers:
      raise ValueError('The trace worker cannot review its own candidate')
    expected = {
      'sol': ('gpt-6.1-sol', 'medium'),
      'argon': ('gemini-4-argon-high', None),
      'caw': (
        'grok-4.7[context=500k,reasoning_effort=xhigh,fast=false]',
        None,
      ),
    }
    if (self.trace.model, self.trace.effort) != expected[self.trace.provider]:
      raise ValueError('Use Argon High, Sol Medium or Grok Extra High 500K')
    if self.providers.astra.model != 'gpt-6-astra':
      raise ValueError('Astra review requires gpt-6-astra')
    if not (self.providers.opus.model or '').startswith('claude-opus-'):
      raise ValueError('Opus review requires an explicit Claude Opus model')
    caps = Limits(
      invocationMs=8 * 60 * 60 * 1000,
      workerMs=2 * 60 * 60 * 1000,
      maxCalls=100,
      maxAttempts=10,
      maxOutputBytes=33554432,
      maxSvgBytes=2097152,
      maxSegments=100000,
      maxDepth=32,
      maxCoordinate=1000000,
      maxRasterMs=30000,
    )
    for name, value in asdict(self.limits).items():
      if not 0 < value <= getattr(caps, name):
        raise ValueError(f'Invalid limit: {name}')
    for tier, cap in [
      (self.exports.shelf, Tier(96, 320, 80)),
      (self.exports.detail, Tier(256, 1280, 84)),
    ]:
      if not (
        0 < tier.width <= cap.width
        and 0 < tier.height <= cap.height
        and tier.quality
        == (cap.quality if self.exports.encoding == 'lossy-webp' else None)
      ):
        raise ValueError('Invalid raster export budget')
    for provider in get_args(Provider):
      if not getattr(self.providers, provider).command:
        raise ValueError('Provider command is empty')


@dataclass
class InputJob:
  id: str
  title: str
  notes: list[str]
  source: Source | None
  reference: Pin | None
  front: Pin | None
  blockedReason: str | None


@dataclass
class Inputs:
  version: Literal[1]
  jobs: list[InputJob]


@dataclass
class Trace:
  jobId: str
  inputFingerprint: str
  status: Literal['candidate', 'blocked_source']
  svg: str
  svgFile: Pin | None
  provenance: list[str]
  unresolved: list[str]


@dataclass
class Categories:
  typography: bool
  marks: bool
  layout: bool
  colour: bool
  photographicDetail: bool
  texture: bool


@dataclass
class Defect:
  region: str
  observation: str
  repair: str


@dataclass
class TierReview:
  sourceCorrespondence: bool
  fullFaceCoverage: bool


@dataclass
class TierReviews:
  shelf: TierReview
  detail: TierReview


@dataclass
class Review:
  jobId: str
  inputFingerprint: str
  candidateSha256: str
  rubricSha256: str
  inspectedImages: list[str]
  verdict: Literal['pass', 'revise', 'blocked_source', 'needs_review']
  categories: Categories
  tiers: TierReviews
  defects: list[Defect]


@dataclass
class Image(Pin):
  width: int
  height: int
  bytes: int


@dataclass
class Candidate:
  master: Pin
  shelf: Image
  detail: Image
  provenance: list[str]
  recipe: str


@dataclass
class Receipt:
  provider: Provider
  invocation: str
  transportSha256: str
  result: Review


@dataclass
class Job:
  input: InputJob
  inputFingerprint: str
  status: Status
  reason: str | None
  attempts: int
  candidate: Candidate | None
  reviews: list[Receipt]


@dataclass
class State:
  version: Literal[1]
  createdAt: str
  run: str
  config: Config
  configPin: Pin
  implementation: str
  sourcePins: list[Pin]
  jobs: list[Job]
  calls: int


@dataclass
class Transport:
  provider: Provider
  complete: bool
  exitCode: int | None
  response: str
  conversationId: str | None
  model: str | None
  error: str | None
  usage: object


def mapping(value: object) -> dict[str, object]:
  if not isinstance(value, dict):
    raise ValueError('Expected a JSON object')
  if any(not isinstance(key, str) for key in value):
    raise ValueError('Expected string object keys')
  return {key: item for key, item in value.items()}


def optional_text(value: object) -> str | None:
  if value is None or isinstance(value, str):
    return value
  raise ValueError('Expected string or null')


def strict_json(text: str) -> object:
  def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
      if key in result:
        raise ValueError(f'Duplicate JSON key: {key}')
      result[key] = value
    return result

  def invalid(value: str) -> object:
    raise ValueError(f'Nonfinite JSON constant: {value}')

  return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def decode(cls: type[T], value: object) -> T:
  """Reject missing/extra fields and primitive coercions at JSON boundaries."""
  if not is_dataclass(cls) or not isinstance(value, dict):
    raise ValueError(f'Expected {cls.__name__} object')
  expected = {field.name for field in fields(cls)}
  if set(value) != expected:
    raise ValueError(f'{cls.__name__} fields differ: {set(value) ^ expected}')
  hints = get_type_hints(cls)
  converted = {
    name: convert(hints[name], item) for name, item in value.items()
  }
  return cls(**converted)


def convert(annotation: object, value: object) -> object:
  origin, args = get_origin(annotation), get_args(annotation)
  if origin is types.UnionType:
    for option in args:
      try:
        return convert(option, value)
      except ValueError:
        pass
  elif origin is Literal:
    if any(type(value) is type(item) and value == item for item in args):
      return value
  elif origin is list and isinstance(value, list):
    return [convert(args[0], item) for item in value]
  elif isinstance(annotation, type) and is_dataclass(annotation):
    return decode(annotation, value)
  elif annotation is object:
    return value
  elif type(value) is annotation:
    if isinstance(value, float) and not math.isfinite(value):
      raise ValueError('Nonfinite JSON number')
    return value
  raise ValueError(f'Invalid value for {annotation}')


def json_schema(annotation: object) -> dict[str, object]:
  origin, args = get_origin(annotation), get_args(annotation)
  if origin is Literal:
    return {
      'enum': list(args),
      'type': 'string' if isinstance(args[0], str) else 'integer',
    }
  if origin is types.UnionType:
    return {'anyOf': [json_schema(item) for item in args]}
  if origin is list:
    return {'type': 'array', 'items': json_schema(args[0])}
  if isinstance(annotation, type) and is_dataclass(annotation):
    hints = get_type_hints(annotation)
    return {
      'type': 'object',
      'properties': {
        field.name: json_schema(hints[field.name])
        for field in fields(annotation)
      },
      'required': [field.name for field in fields(annotation)],
      'additionalProperties': False,
    }
  primitives: dict[type, str] = {
    str: 'string',
    int: 'integer',
    bool: 'boolean',
    type(None): 'null',
  }
  if isinstance(annotation, type) and annotation in primitives:
    return {'type': primitives[annotation]}
  raise ValueError(f'Unsupported schema type: {annotation}')
