"""Simple HTTP server: serves the game files + provides /ai_move endpoint.

Usage:
    cd ~/geometry-game/ai
    python3 server.py                  # MCTS + value net (default)
    python3 server.py --ai greedy      # greedy best-immediate-score
    python3 server.py 8080             # custom port
    python3 server.py 8080 --ai greedy # both

Then open http://localhost:8765/ in a browser (Windows browser works from WSL).
Toggle "vs AI" in the sidebar to play against the agent as Red.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planespace_env import State
from macro_mcts import MacroMCTS
from greedy_ai import GreedyAI

GAME_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

_VALUE_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'macro_value', 'value_net.pt')


def _build_mcts():
    value_net = None
    value_blend = 0.0
    if os.path.exists(_VALUE_CKPT):
        import torch
        from network import PlanespaceNet
        value_net = PlanespaceNet()
        value_net.load_state_dict(torch.load(_VALUE_CKPT, map_location='cpu'))
        value_net.eval()
        value_blend = 1.0
        print(f'Loaded value network from {_VALUE_CKPT}')
    else:
        print('No value network found — using pure rollout evaluation')
    return MacroMCTS(n_sims=200, n_cands=20, value_net=value_net, value_blend=value_blend)


def _build_agent(ai_mode: str):
    if ai_mode == 'greedy':
        print('AI mode: greedy (highest immediate score)')
        return GreedyAI()
    print('AI mode: MCTS')
    return _build_mcts()


def _agent_choose(agent, state: State):
    if isinstance(agent, GreedyAI):
        return agent.choose(state)
    return agent.choose(state, temperature=0.0)


# Parsed from argv below
_ai_mode = 'greedy' if '--ai' in sys.argv and sys.argv[sys.argv.index('--ai') + 1] == 'greedy' else 'mcts'
agent = _build_agent(_ai_mode)

STATIC = {
    '/':            ('index.html', 'text/html'),
    '/index.html':  ('index.html', 'text/html'),
    '/replay.html': ('replay.html', 'text/html'),
    '/replay.json': ('replay.json', 'application/json'),
}


def _parse_state(data: dict) -> State:
    """Convert the JSON game state sent from index.html into a Python State."""
    shapes = tuple(
        (tuple((v['x'], v['y']) for v in sh['verts']), sh['player'])
        for sh in data.get('shapes', [])
    )
    claimed_gems = frozenset()
    for gk in data.get('claimedGems', []):
        x, y = map(int, gk.split(','))
        claimed_gems = claimed_gems | {(x, y)}
    scores = tuple(data.get('scores', [0, 0]))
    player = int(data.get('player', 0))
    passes = int(data.get('passes', 0))
    return State(
        shapes=shapes,
        claimed_gems=claimed_gems,
        scores=scores,
        player=player,
        sel=(),
        passes=passes,
        done=False,
        plies=len(shapes),
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request logging

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, rel_path, content_type):
        full = os.path.join(GAME_ROOT, rel_path)
        try:
            with open(full, 'rb') as f:
                data = f.read()
        except FileNotFoundError:
            self.send_error(404, 'Not found')
            return
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path in STATIC:
            rel, ctype = STATIC[self.path]
            self._send_file(rel, ctype)
        else:
            self.send_error(404, 'Not found')

    def do_POST(self):
        if self.path != '/ai_move':
            self.send_error(404)
            return
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
            state = _parse_state(data)
            pts = _agent_choose(agent, state)
            if pts is None:
                self._send_json({'pass': True})
            else:
                self._send_json({'verts': [[p[0], p[1]] for p in pts]})
        except Exception as e:
            self._send_json({'error': str(e)}, status=500)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--') and a != 'greedy']
    port = int(args[0]) if args else 8765
    server = HTTPServer(('', port), Handler)
    print(f'Planespace AI server running at http://localhost:{port}/')
    print(f'  Toggle "vs AI" in the sidebar to play against the agent.')
    print(f'  Press Ctrl+C to stop.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nServer stopped.')


if __name__ == '__main__':
    main()
