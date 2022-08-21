# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

from __future__ import annotations

import socket, sys, threading, json, os, argparse, logging, signal
from logging.handlers import RotatingFileHandler


def setup_logger(log_level:int=logging.INFO):
  logger = logging.getLogger(__name__)
  logger.setLevel(log_level)

  console_handler = logging.StreamHandler()
  console_handler.setLevel(log_level)

  # Create file handler with rotation (max 5MB, keep 3 backups)
  file_handler = RotatingFileHandler('file_synchronizer.log', maxBytes=5*1024*1024, backupCount=3)
  file_handler.setLevel(log_level)

  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  console_handler.setFormatter(formatter)
  file_handler.setFormatter(formatter)

  logger.addHandler(console_handler)
  logger.addHandler(file_handler)

  return logger

logger = setup_logger()

def validate_ip(s:str)->bool:
  try: return len((a := s.split('.'))) == 4 and all(x.isdigit() and 0 <= int(x) <= 255 for x in a)
  except: return False

def validate_port(x:str)->bool:return x.isdigit() and 0 <= int(x) <= 65535

def get_file_info(ignored:tuple[str,...]=('.so','.py','.dll'))->list[dict[str,str|int]]:
  return [
    {'name': filename, 'mtime': int(os.path.getmtime(filename))}
    for filename in os.listdir('.')
    if os.path.isfile(filename) and not filename.endswith(ignored)
  ]


def check_port_available(check_port:int)->bool:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
      s.bind(('0.0.0.0', check_port))
      return True
    except socket.error:
      return False


def get_next_available_port(initial_port:int)->int|bool:
  port = initial_port
  while port <= 65535:
    if check_port_available(port): return port
    port += 1
  return False


class FileSynchronizer(threading.Thread):
  def __init__(self,trackerhost:int,trackerport:int,port:int,host:str='0.0.0.0'):
    threading.Thread.__init__(self)
    self.daemon = True

    # Own port and IP address for serving file requests to other peers
    self.port = port
    self.host = host

    # Tracker IP/hostname and port
    self.trackerhost = trackerhost
    self.trackerport = trackerport

    self.BUFFER_SIZE = 8192

    # Flag to control the thread execution
    self.running = True

    # Timer for sync operations
    self.sync_timer: threading.Timer | None = None

    # Create a TCP socket to communicate with the tracker
    self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.client.settimeout(180)

    # Store the message to be sent to the tracker.
    # Initialize to the Init message that contains port number and file info.
    # Refer to Table 1 in Instructions.pdf for the format of the Init message
    # You can use json.dumps to convert a python dictionary to a json string
    # Encode using UTF-8
    self.msg = json.dumps(dict(port=self.port, files=get_file_info())).encode()

    # Create a TCP socket to serve file requests from peers.
    self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
      self.server.bind((self.host, self.port))
    except socket.error:
      logger.error('Bind failed %s', socket.error)
      sys.exit()
    self.server.listen(10)
    self.server.settimeout(1)  # Set timeout to allow checking running flag

  # Ensure sockets are closed on disconnect
  def exit(self):
    """Gracefully shut down the synchronizer"""
    logger.info("Shutting down file synchronizer...")
    self.running = False

    # Cancel any pending sync timer
    if self.sync_timer and self.sync_timer.is_alive():self.sync_timer.cancel()

    # Close sockets
    try:
      self.server.close()
    except Exception as e:
      logger.error("Error closing server socket: %s", e)

    try:
      self.client.close()
    except Exception as e:
      logger.error("Error closing client socket: %s", e)

    logger.info("File synchronizer shutdown complete")

  def __del__(self): self.exit()

  # Handle file request from a peer(i.e., send the file content to peers)
  def process_message(self,conn:socket.socket,addr:str):
    file_name=''
    try:
      file_name = conn.recv(self.BUFFER_SIZE).decode('utf-8')
      logger.info('Received file request: %s from %s', file_name, addr)
      with open(file_name, 'rb') as file: file_content = file.read()
      conn.sendall(file_content)
      logger.info('Sent %d bytes of %s to %s', len(file_content), file_name, addr)
    except FileNotFoundError:
      logger.error('File not found: %s', file_name)
      conn.sendall(b'')
    except Exception as e:
      logger.error('Error processing request: %s', e)
    finally:
      conn.close()

  def run(self):
    try:
      self.client.connect((self.trackerhost, self.trackerport))
      self.sync_timer = threading.Timer(2, self.sync)
      self.sync_timer.daemon = True
      self.sync_timer.start()
      logger.info('Waiting for connections on port %s', self.port)

      while self.running:
        try:
          conn, addr = self.server.accept()
          client_thread = threading.Thread(target=self.process_message, args=(conn, addr))
          client_thread.daemon = True
          client_thread.start()
        except socket.timeout:
          # This is expected due to the socket timeout we set
          continue
        except Exception as e:
          if self.running:  # Only log if we're still supposed to be running
            logger.error("Error accepting connection: %s", e)

      logger.info("File synchronizer main loop exited")
    except Exception as e:
      logger.error("Error in synchronizer main loop: %s", e)

  # Send Init or KeepAlive message to tracker, handle directory response message
  # and request files from peers
  def sync(self):
    if not self.running: return

    try:
      logger.info('Connecting to: %s:%s', self.trackerhost, self.trackerport)
      # Step 1. send Init msg to tracker (Note init msg only sent once)
      self.client.sendall(self.msg)

      # Step 2. now receive a directory response message from tracker
      directory_response_message = ''
      data = self.client.recv(self.BUFFER_SIZE)
      directory_response_message = data.decode('utf-8')

      logger.debug('Received from tracker: %s', directory_response_message)

      # Step 3. parse the directory response message. If it contains new or
      # more up-to-date files, request the files from the respective peers.
      if directory_response_message:
        try:
          # Parse the tracker response
          directory = json.loads(directory_response_message)

          # Get information about local files
          local_files = {}
          for file_info in get_file_info():
            local_files[file_info['name']] = file_info['mtime']

          # Check for new or more up-to-date files
          for file_name, file_info in directory.items():
            if not self.running:
              break

            file_ip = file_info['ip']
            file_port = file_info['port']
            file_mtime = file_info['mtime']

            # Check if file is new or more recent than our local copy
            if file_name not in local_files or file_mtime > local_files[file_name]:
              logger.info('Requesting file %s from %s:%s', file_name, file_ip, file_port)
              peer_socket = None
              try:
                # Create a socket to connect to the peer
                peer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                peer_socket.settimeout(10)  # Set a timeout of 10 seconds

                # Connect to the peer
                peer_socket.connect((file_ip, file_port))

                # Send the file name as request
                peer_socket.sendall(file_name.encode('utf-8'))

                # Receive the file content
                file_content = b''
                while True:
                  chunk = peer_socket.recv(self.BUFFER_SIZE)
                  if not chunk:
                    break
                  file_content += chunk

                # Write the file to disk
                with open(file_name, 'wb') as file:
                  file.write(file_content)

                # Set the modified time to match the timestamp in the directory
                os.utime(file_name, (file_mtime, file_mtime))

                logger.info('Downloaded file %s from %s:%s', file_name, file_ip, file_port)

              except (socket.error, socket.timeout) as e:
                logger.error('Error connecting to peer %s:%s: %s', file_ip, file_port, e)
                # Skip this file and proceed to the next
              finally:
                # Close the connection
                if peer_socket:
                  peer_socket.close()

        except json.JSONDecodeError as e:
          logger.error('Error decoding tracker response: %s', e)

      self.msg = json.dumps({'port': self.port}).encode('utf-8')

      # Schedule next sync if still running
      if self.running:
        self.sync_timer = threading.Timer(5, self.sync)
        self.sync_timer.daemon = True
        self.sync_timer.start()

    except Exception as e:
      logger.error("Error in sync operation: %s", e)
      # Still try to schedule next sync if running
      if self.running:
        self.sync_timer = threading.Timer(5, self.sync)
        self.sync_timer.daemon = True
        self.sync_timer.start()

def signal_handler(sig:int, frame):
    logger.info("Received signal %s, shutting down...", sig)
    if 'synchronizer' in globals(): globals()['synchronizer'].exit()
    sys.exit(0)

def main()->int:
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description='File Synchronizer Client')
    parser.add_argument('tracker_ip', help='Tracker server IP address')
    parser.add_argument('tracker_port', help='Tracker server port number')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default='INFO', help='Set logging level')

    args = parser.parse_args()

    # Calculate IP header checksum
    data = "4510 003c 1c46 4501 4006 b1e6 ac10 0a63"
    # Remove spaces and convert to bytes
    hex_data = data.replace(" ", "")

    # Process 16 bits (2 bytes) at a time
    sum = 0
    # Process each 16-bit word (4 hex digits)
    for i in range(0, len(hex_data), 4):
        if i+3 < len(hex_data):  # Ensure we have a full 16-bit word
            word = int(hex_data[i:i+4], 16)
            sum += word
            # Handle overflow beyond 16 bits
            if sum > 0xFFFF:
                sum = (sum & 0xFFFF) + (sum >> 16)

    # Take one's complement
    checksum = (~sum) & 0xFFFF

    print(f"IP Header Checksum: 0x{checksum:04x}")

    # Set log level if specified
    if args.log_level:
        logger.setLevel(getattr(logging, args.log_level))
        for handler in logger.handlers:
            handler.setLevel(getattr(logging, args.log_level))

    # Validate IP and port
    if not validate_ip(args.tracker_ip):
        parser.error('Invalid tracker IP address')

    if not validate_port(args.tracker_port):
        parser.error('Invalid tracker port number')

    tracker_ip = args.tracker_ip
    tracker_port = int(args.tracker_port)

    # Get free port
    synchronizer_port = get_next_available_port(8000)
    if not synchronizer_port:
        logger.critical("No available ports found")
        return 1

    logger.info("Starting file synchronizer on port %s", synchronizer_port)
    global synchronizer
    synchronizer = FileSynchronizer(tracker_ip, tracker_port, synchronizer_port)
    synchronizer.start()

    try:
        # Keep the main thread alive to handle signals
        while synchronizer.is_alive():
            synchronizer.join(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        synchronizer.exit()
    except Exception as e:
        logger.error("Error in main thread: %s", e)
        synchronizer.exit()
        return 1

    return 0

if __name__=='__main__':main()
