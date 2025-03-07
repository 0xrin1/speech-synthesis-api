"""Client module for the Speech API."""

from src.client.base_client import BaseClient
from src.client.cli_client import CLIClient
from src.client.playback_client import PlaybackClient

__all__ = ['BaseClient', 'CLIClient', 'PlaybackClient']