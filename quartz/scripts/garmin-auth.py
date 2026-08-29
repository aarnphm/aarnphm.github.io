import argparse
import getpass
import json
import re
import sys
from pathlib import Path

from garminconnect import Garmin


ROOT = Path(__file__).resolve().parents[2]
TOKENSTORE = ROOT / 'quartz' / '.quartz-cache' / 'garmin-auth'
TOKEN_FILE = TOKENSTORE / 'garmin_tokens.json'
QUERY_VALUE = re.compile(r'([?&][\w.-]+=)[^&\s)\'\"]+')
BEARER_VALUE = re.compile(r'(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+')


def safe_error(error: Exception) -> str:
  message = QUERY_VALUE.sub(r'\1<redacted>', str(error))
  return BEARER_VALUE.sub(r'\1<redacted>', message)


def bearer_headers(garmin: Garmin) -> dict[str, str]:
  headers = garmin.client.get_api_headers()
  authorization = headers.get('Authorization')
  if not authorization or not authorization.startswith('Bearer '):
    raise RuntimeError('Garmin did not issue a bearer session')
  if any(name.casefold() == 'cookie' for name in headers):
    raise RuntimeError(
      'Garmin returned a cookie session instead of bearer authentication'
    )
  return headers


def authorize() -> None:
  email = input('Garmin email: ').strip()
  if not email:
    raise RuntimeError('Garmin email is required')
  password = getpass.getpass('Garmin password: ')
  if not password:
    raise RuntimeError('Garmin password is required')

  def prompt_mfa() -> str:
    code = getpass.getpass('Garmin MFA code: ').strip()
    if not code:
      raise RuntimeError('Garmin MFA code is required')
    return code

  garmin = Garmin(email=email, password=password, prompt_mfa=prompt_mfa)
  garmin.login(str(TOKENSTORE))
  bearer_headers(garmin)
  garmin.client.dump(str(TOKENSTORE))
  print(f'Garmin authorization stored in {TOKEN_FILE}')


def session() -> None:
  if not TOKEN_FILE.is_file():
    raise RuntimeError('Garmin authorization is missing; run pnpm garmin:auth')
  garmin = Garmin()
  garmin.login(str(TOKENSTORE))
  headers = bearer_headers(garmin)
  garmin.client.dump(str(TOKENSTORE))
  sys.stdout.write(json.dumps({'headers': headers}, separators=(',', ':')))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('action', choices=('authorize', 'session'))
  args = parser.parse_args()
  if args.action == 'authorize':
    authorize()
  else:
    session()


if __name__ == '__main__':
  try:
    main()
  except Exception as error:
    print(f'[garmin-auth] failed: {safe_error(error)}', file=sys.stderr)
    raise SystemExit(1) from None
