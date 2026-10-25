"""Observe the runner's model processes through their loopback HTTP APIs."""

from collections import deque
import json
import time
from urllib.error import URLError
from urllib.request import urlopen

from inputs import mapping, number


class Models:
  def __init__(self) -> None:
    self.samples: dict[tuple[str, int], deque[tuple[float, int]]] = {}
    self.saved: dict[tuple[str, int], tuple[float, dict[str, object]]] = {}

  def observe(
    self, port: str, record: dict[str, object], alive: bool
  ) -> dict[str, object]:
    pid = number(record.get('pid'))
    result: dict[str, object] = {
      'port': port,
      'pid': pid,
      'alive': alive,
      'healthy': False,
      'model': record.get('model', record.get('repo')),
    }
    if (
      (not alive and record.get('owned') is not False)
      or not port.isdecimal()
      or not 1 <= int(port) <= 65535
    ):
      return result
    key, now = (port, pid), time.monotonic()
    previous = self.saved.get(key)
    if previous and now - previous[0] < 3:
      return previous[1]

    def read(endpoint: str) -> dict[str, object]:
      with urlopen(
        f'http://127.0.0.1:{port}/{endpoint}', timeout=0.75
      ) as response:
        return mapping(json.load(response))

    try:
      health = read('health')
      if record.get('repo') == 'mlx-community/embeddinggemma-2-bf16':
        result['healthy'] = health.get('status') == 'ok' and all(
          health.get(name) == record.get(name) for name in ('repo', 'revision')
        )
      else:
        identity = read('ocr/identity')
        result['healthy'] = health.get('status') == 'healthy' and all(
          identity.get(name) == record.get(name)
          for name in ('model', 'revision', 'serverRevision')
        )
        metrics = read('metrics')
        summary, server = (
          mapping(metrics.get('summary')),
          mapping(metrics.get('server')),
        )
        samples = self.samples.setdefault(key, deque(maxlen=11))
        tokens = number(summary.get('completion_tokens_total'))
        if samples and tokens < samples[-1][1]:
          samples.clear()
        samples.append((now, tokens))
        while len(samples) > 1 and now - samples[0][0] > 30:
          samples.popleft()
        elapsed = now - samples[0][0]
        result.update(
          inFlight=summary.get('in_flight'),
          maxSequences=identity.get('maxNumSeqs'),
          queued=server.get('request_queue_depth'),
          completedRequests=summary.get('requests_completed'),
          failedRequests=summary.get('requests_failed'),
          completionTokens=tokens,
          completedTokensPerSecond=(tokens - samples[0][1]) / elapsed
          if elapsed >= 10
          else None,
          sampleSeconds=elapsed,
          continuousBatching=server.get('continuous_batching_enabled'),
          prefixCache=mapping(server.get('apc')).get('enabled'),
          prefixHits=number(mapping(server.get('apc')).get('exact_hits'))
          + number(mapping(server.get('apc')).get('lookups_hit')),
          thinkingEnabled=mapping(metrics.get('latest')).get(
            'thinking_enabled'
          ),
          peakMemoryGB=mapping(metrics.get('latest')).get('peak_memory_gb'),
        )
    except (OSError, URLError, ValueError) as error:
      result.update(healthy=False, error=str(error))
    result['owned'] = record.get('owned', True)
    self.saved[key] = (now, result)
    return result
