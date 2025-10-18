#!/usr/bin/env python3
"""
Piet Programming Language Interpreter
"""

from PIL import Image
from collections import deque
import sys
import os

# Piet color palette (18 colors + black/white)
PIET_PALETTE = {
    (255,192,192):0,  (255,0,0):1,     (192,0,0):2,
    (255,255,192):3,  (255,255,0):4,   (192,192,0):5,
    (192,255,192):6,  (0,255,0):7,     (0,192,0):8,
    (192,255,255):9,  (0,255,255):10,  (0,192,192):11,
    (192,192,255):12, (0,0,255):13,    (0,0,192):14,
    (255,192,255):15, (255,0,255):16,  (192,0,192):17
}

BLACK = -1
WHITE = -2

# Direction pointer vectors: right, down, left, up
DP_VECS = [(1,0), (0,1), (-1,0), (0,-1)]

class PietInterpreter:
    def __init__(self, image_path):
        """Load and parse Piet image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to open image: {e}")
        
        self.w, self.h = img.size
        pixels = list(img.getdata())
        
        # Build color grid
        self.grid = [[None]*self.w for _ in range(self.h)]
        for y in range(self.h):
            for x in range(self.w):
                rgb = tuple(pixels[y*self.w + x])
                if rgb == (0,0,0):
                    self.grid[y][x] = BLACK
                elif rgb == (255,255,255):
                    self.grid[y][x] = WHITE
                else:
                    self.grid[y][x] = PIET_PALETTE.get(rgb, WHITE)
        
        # Build regions via flood fill
        self.regions, self.region_id = self._build_regions()
        
        # Execution state
        self.dp = 0  # Direction pointer: 0=right, 1=down, 2=left, 3=up
        self.cc = 0  # Codel chooser: 0=left, 1=right
        self.stack = []
        self.current_region = self._find_start_region()
    
    def _build_regions(self):
        """Build connected regions using flood fill"""
        visited = [[False]*self.w for _ in range(self.h)]
        region_id = [[-1]*self.w for _ in range(self.h)]
        regions = {}
        rid = 0
        
        for y in range(self.h):
            for x in range(self.w):
                if visited[y][x]:
                    continue
                
                color = self.grid[y][x]
                queue = [(x, y)]
                pixels = []
                visited[y][x] = True
                
                # Flood fill
                while queue:
                    cx, cy = queue.pop()
                    pixels.append((cx, cy))
                    
                    for nx, ny in self._neighbors(cx, cy):
                        if not visited[ny][nx] and self.grid[ny][nx] == color:
                            visited[ny][nx] = True
                            queue.append((nx, ny))
                
                regions[rid] = {"color": color, "pixels": pixels}
                for px, py in pixels:
                    region_id[py][px] = rid
                rid += 1
        
        return regions, region_id
    
    def _neighbors(self, x, y):
        """Get orthogonal neighbors"""
        for dx, dy in DP_VECS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                yield nx, ny
    
    def _find_start_region(self):
        """Find first non-black region (top-left scan)"""
        for y in range(self.h):
            for x in range(self.w):
                if self.grid[y][x] != BLACK:
                    return self.region_id[y][x]
        return 0
    
    def _choose_edge_codel(self, pixels, dp, cc):
        """Select edge codel based on DP and CC"""
        dx, dy = DP_VECS[dp]
        
        # Find codels furthest in DP direction
        best_dp = None
        candidates = []
        for x, y in pixels:
            val = x*dx + y*dy
            if best_dp is None or val > best_dp:
                best_dp = val
                candidates = [(x, y)]
            elif val == best_dp:
                candidates.append((x, y))
        
        # Among candidates, choose by CC (perpendicular to DP)
        cc_left = (-dy, dx)
        cc_right = (dy, -dx)
        cc_vec = cc_left if cc == 0 else cc_right
        
        best_cc = None
        chosen = candidates[0]
        for x, y in candidates:
            val = x*cc_vec[0] + y*cc_vec[1]
            if best_cc is None or val > best_cc:
                best_cc = val
                chosen = (x, y)
        
        return chosen
    
    def _get_command(self, old_color, new_color):
        """Determine command from color transition"""
        if old_color < 0 or new_color < 0:
            return None
        
        old_h, old_l = old_color // 3, old_color % 3
        new_h, new_l = new_color // 3, new_color % 3
        dh = (new_h - old_h) % 6
        dl = (new_l - old_l) % 3
        
        if dh == 0 and dl == 0:
            return None
        
        commands = {
            0: {1: "push", 2: "pop"},
            1: {0: "add", 1: "subtract", 2: "multiply"},
            2: {0: "divide", 1: "mod", 2: "not"},
            3: {0: "greater", 1: "pointer", 2: "switch"},
            4: {0: "duplicate", 1: "roll", 2: "in_number"},
            5: {0: "in_char", 1: "out_number", 2: "out_char"},
        }
        
        return commands.get(dh, {}).get(dl, None)
    
    def _execute(self, cmd, block_size):
        """Execute Piet command"""
        if cmd == "push":
            self.stack.append(block_size)
        elif cmd == "pop":
            self._pop()
        elif cmd == "add":
            a, b = self._pop(), self._pop()
            self.stack.append(b + a)
        elif cmd == "subtract":
            a, b = self._pop(), self._pop()
            self.stack.append(b - a)
        elif cmd == "multiply":
            a, b = self._pop(), self._pop()
            self.stack.append(b * a)
        elif cmd == "divide":
            a, b = self._pop(), self._pop()
            self.stack.append(0 if a == 0 else b // a)
        elif cmd == "mod":
            a, b = self._pop(), self._pop()
            self.stack.append(0 if a == 0 else b % a)
        elif cmd == "not":
            a = self._pop()
            self.stack.append(0 if a else 1)
        elif cmd == "greater":
            a, b = self._pop(), self._pop()
            self.stack.append(1 if b > a else 0)
        elif cmd == "pointer":
            a = self._pop()
            self.dp = (self.dp + (a % 4)) % 4
        elif cmd == "switch":
            a = self._pop()
            if a % 2 == 1:
                self.cc = 1 - self.cc
        elif cmd == "duplicate":
            if self.stack:
                self.stack.append(self.stack[-1])
        elif cmd == "roll":
            x, y = self._pop(), self._pop()
            if 0 < y <= len(self.stack):
                r = x % y
                if r:
                    top = self.stack[-y:]
                    self.stack[-y:] = top[-r:] + top[:-r]
        elif cmd == "in_number":
            self.stack.append(0)  # Stub for input
        elif cmd == "in_char":
            self.stack.append(0)  # Stub for input
        elif cmd == "out_number":
            print(self._pop(), end='')
        elif cmd == "out_char":
            val = self._pop()
            print(chr(val % 256), end='', flush=True)
    
    def _pop(self):
        """Pop from stack, return 0 if empty"""
        return self.stack.pop() if self.stack else 0
    
    def _traverse_white(self, start_x, start_y):
        """Traverse white block and find exit"""
        dx, dy = DP_VECS[self.dp]
        
        # Find all white codels connected to start
        queue = deque([(start_x, start_y)])
        seen = {(start_x, start_y)}
        white_codels = []
        
        while queue:
            x, y = queue.popleft()
            white_codels.append((x, y))
            
            for nx, ny in self._neighbors(x, y):
                if (nx, ny) not in seen and self.grid[ny][nx] == WHITE:
                    seen.add((nx, ny))
                    queue.append((nx, ny))
        
        # Find exit candidates (white codels adjacent to color blocks)
        exits = []
        for wx, wy in white_codels:
            ex, ey = wx + dx, wy + dy
            if 0 <= ex < self.w and 0 <= ey < self.h and self.grid[ey][ex] >= 0:
                exits.append((wx, wy, ex, ey))
        
        if not exits:
            return None
        
        # Choose exit based on DP and CC
        cc_left = (-dy, dx)
        cc_right = (dy, -dx)
        cc_vec = cc_left if self.cc == 0 else cc_right
        
        best_dp, best_cc = None, None
        chosen = None
        
        for wx, wy, ex, ey in exits:
            val_dp = wx*dx + wy*dy
            val_cc = wx*cc_vec[0] + wy*cc_vec[1]
            
            if best_dp is None or val_dp > best_dp or \
               (val_dp == best_dp and val_cc > best_cc):
                best_dp, best_cc = val_dp, val_cc
                chosen = (wx, wy, ex, ey)
        
        return chosen[2:] if chosen else None  # Return exit coords
    
    def run(self, max_steps=200000):
        """Execute Piet program"""
        step = 0
        
        try:
            while step < max_steps:
                step += 1
                region = self.regions[self.current_region]
                color = region["color"]
                block_size = len(region["pixels"])
                
                edge_x, edge_y = self._choose_edge_codel(region["pixels"], self.dp, self.cc)
                moved = False
                
                # Try to move (up to 8 attempts)
                for attempt in range(8):
                    dx, dy = DP_VECS[self.dp]
                    nx, ny = edge_x + dx, edge_y + dy
                    
                    # Check bounds and black
                    if not (0 <= nx < self.w and 0 <= ny < self.h) or \
                       self.grid[ny][nx] == BLACK:
                        # Rotate CC/DP and recalculate edge
                        if attempt % 2 == 0:
                            self.cc = 1 - self.cc
                        else:
                            self.dp = (self.dp + 1) % 4
                        edge_x, edge_y = self._choose_edge_codel(region["pixels"], self.dp, self.cc)
                        continue
                    
                    # Direct color block
                    if self.grid[ny][nx] >= 0:
                        next_region = self.region_id[ny][nx]
                        cmd = self._get_command(color, self.grid[ny][nx])
                        if cmd:
                            self._execute(cmd, block_size)
                        self.current_region = next_region
                        moved = True
                        break
                    
                    # White block traversal
                    if self.grid[ny][nx] == WHITE:
                        result = self._traverse_white(nx, ny)
                        
                        if result:
                            ex, ey = result
                            next_region = self.region_id[ey][ex]
                            cmd = self._get_command(color, self.grid[ey][ex])
                            if cmd:
                                self._execute(cmd, block_size)
                            self.current_region = next_region
                            moved = True
                            break
                        else:
                            # White traversal blocked
                            if attempt % 2 == 0:
                                self.cc = 1 - self.cc
                            else:
                                self.dp = (self.dp + 1) % 4
                            edge_x, edge_y = self._choose_edge_codel(region["pixels"], self.dp, self.cc)
                            continue
                
                if not moved:
                    break  # Program terminates
        
        except KeyboardInterrupt:
            print("\n[Interrupted by user]", file=sys.stderr)
            sys.exit(130)
        except Exception as e:
            print(f"\n[Runtime error: {e}]", file=sys.stderr)
            sys.exit(1)
        
        print()  # Final newline

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <piet_image.png>", file=sys.stderr)
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        interpreter = PietInterpreter(image_path)
        interpreter.run()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[Interrupted]", file=sys.stderr)
        sys.exit(130)