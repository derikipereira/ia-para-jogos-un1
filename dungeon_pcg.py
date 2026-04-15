"""
PCG Dungeon: construtor (salas + corredores em árvore), crítico (BFS em espaço de estados)
e renderização com folha de sprites 48x48 (9 tiles de 16x16).

Regras principais: uma porta por aresta da árvore; folhas com tesouro sem brechas; monstro
só em corredor; poção obrigatória antes de cada monstro (estado may_fight na BFS); poções ≥
monstros. Visual: parede = tile (0,0); preenchimento não andável = VOID = 3.º tile da 1.ª
linha (2,0); piso = (2,2). Ver também o notebook PCG_Dungeon.ipynb.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from PIL import Image, ImageEnhance

Vec2 = Tuple[int, int]


class Terrain(IntEnum):
    VOID = 0
    WALL = 1
    FLOOR = 2
    DOOR = 3  # fechada (bloqueia); aberta vira FLOOR na simulação


class EntityKind(IntEnum):
    NONE = 0
    MONSTER = 1
    POTION = 2
    KEY = 3
    TREASURE = 4


@dataclass
class Room:
    x: int
    y: int
    w: int
    h: int
    id: int

    def cells(self) -> Set[Vec2]:
        s: Set[Vec2] = set()
        for yy in range(self.y, self.y + self.h):
            for xx in range(self.x, self.x + self.w):
                s.add((xx, yy))
        return s

    def center(self) -> Vec2:
        return (self.x + self.w // 2, self.y + self.h // 2)


@dataclass
class DungeonMap:
    width: int
    height: int
    terrain: List[List[int]]  # Terrain
    entity: List[List[int]]  # EntityKind
    entity_index: List[List[int]]  # id local por tipo (0..n-1) ou -1
    start: Vec2
    exit: Vec2
    door_cells: List[Vec2] = field(default_factory=list)
    door_ids: Dict[Vec2, int] = field(default_factory=dict)
    monster_cells: List[Vec2] = field(default_factory=list)
    potion_cells: List[Vec2] = field(default_factory=list)
    key_cells: List[Vec2] = field(default_factory=list)
    treasure_cells: List[Vec2] = field(default_factory=list)
    corridor_cells: Set[Vec2] = field(default_factory=set)
    room_cells: Set[Vec2] = field(default_factory=set)


def _manhattan(a: Vec2, b: Vec2) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _leaf_single_door_exit(leaf: Room, door_cell: Vec2, terrain: List[List[int]], width: int, height: int) -> bool:
    """Sala-folha com tesouro: toda ligação FLOOR/DOOR ao exterior do retângulo deve ser só a porta (sem brechas)."""
    ex: Set[Vec2] = set()
    for c in leaf.cells():
        cx, cy = c
        if int(terrain[cy][cx]) != int(Terrain.FLOOR):
            continue
        for nx, ny in _neigh4(c):
            if leaf.x <= nx < leaf.x + leaf.w and leaf.y <= ny < leaf.y + leaf.h:
                continue
            if not _in_bounds(width, height, (nx, ny)):
                continue
            t = int(terrain[ny][nx])
            if t == int(Terrain.FLOOR) or t == int(Terrain.DOOR):
                ex.add((nx, ny))
    return ex == {door_cell}


def _mark_exterior_void(terrain: List[List[int]], width: int, height: int) -> None:
    """
    Mantém parede (0,0) só no limiar da dungeon. O resto do não-andável passa a VOID,
    renderizado com o 3.º tile da 1.ª linha da folha (coluna 2, linha 0).
    """
    walk = (int(Terrain.FLOOR), int(Terrain.DOOR))
    for y in range(height):
        for x in range(width):
            if int(terrain[y][x]) != int(Terrain.WALL):
                continue
            touches = False
            for nx, ny in _neigh4((x, y)):
                if not _in_bounds(width, height, (nx, ny)):
                    continue
                if int(terrain[ny][nx]) in walk:
                    touches = True
                    break
            if not touches:
                terrain[y][x] = Terrain.VOID


def _l_path(a: Vec2, b: Vec2) -> List[Vec2]:
    """Caminho em L (horizontal depois vertical), inclusivo."""
    cells: List[Vec2] = []
    x0, y0 = a
    x1, y1 = b
    cx, cy = x0, y0
    cells.append((cx, cy))
    while cx != x1:
        cx += 1 if cx < x1 else -1
        cells.append((cx, cy))
    while cy != y1:
        cy += 1 if cy < y1 else -1
        cells.append((cx, cy))
    return cells


def _in_bounds(w: int, h: int, p: Vec2) -> bool:
    return 0 <= p[0] < w and 0 <= p[1] < h


def _neigh4(p: Vec2) -> Iterable[Vec2]:
    x, y = p
    yield x + 1, y
    yield x - 1, y
    yield x, y + 1
    yield x, y - 1


def random_tree_edges(n: int, rng: random.Random) -> List[Tuple[int, int]]:
    """Árvore uniforme aleatória: cada sala j>0 liga a um i<j (método construtivo simples)."""
    edges: List[Tuple[int, int]] = []
    for j in range(1, n):
        i = rng.randint(0, j - 1)
        edges.append((i, j))
    return edges


def try_place_rooms(
    width: int, height: int, n_rooms: int, min_w: int, max_w: int, min_h: int, max_h: int, margin: int
) -> Optional[List[Room]]:
    rooms: List[Room] = []
    attempts = 0
    while len(rooms) < n_rooms and attempts < 4000:
        attempts += 1
        w = random.randint(min_w, max_w)
        h = random.randint(min_h, max_h)
        x = random.randint(margin, width - w - margin)
        y = random.randint(margin, height - h - margin)
        cand = Room(x, y, w, h, len(rooms))
        ok = True
        for r in rooms:
            if not (
                cand.x + cand.w + margin <= r.x
                or r.x + r.w + margin <= cand.x
                or cand.y + cand.h + margin <= r.y
                or r.y + r.h + margin <= cand.y
            ):
                ok = False
                break
        if ok:
            rooms.append(cand)
    if len(rooms) < n_rooms:
        return None
    return rooms


def build_dungeon(
    width: int = 72,
    height: int = 52,
    n_rooms: int = 7,
    monster_density: float = 0.12,
    rng: Optional[random.Random] = None,
) -> Optional[DungeonMap]:
    rng = rng or random.Random()
    rooms = try_place_rooms(width, height, n_rooms, 6, 11, 5, 9, 2)
    if not rooms:
        return None
    edges = random_tree_edges(len(rooms), rng)
    room_cells: Set[Vec2] = set()
    for r in rooms:
        room_cells |= r.cells()
    path_cells: Set[Vec2] = set()
    for i, j in edges:
        p = _l_path(rooms[i].center(), rooms[j].center())
        for c in p:
            path_cells.add(c)
    floor_cells = set(room_cells) | path_cells
    floor_cells = {c for c in floor_cells if _in_bounds(width, height, c)}
    corridor_cells = {c for c in path_cells if c not in room_cells}
    # A topologia em árvore é garantida no grafo de salas (MST); o piso de cada sala
    # retangular contém ciclos locais (vários caminhos dentro da mesma sala), o que é
    # esperado — não exigimos que o grid 4-vizinho inteiro seja uma árvore.

    terrain = [[Terrain.WALL for _ in range(width)] for _ in range(height)]
    entity = [[0 for _ in range(width)] for _ in range(height)]
    entity_index = [[-1 for _ in range(width)] for _ in range(height)]
    for c in floor_cells:
        terrain[c[1]][c[0]] = Terrain.FLOOR

    # Grafo de salas (árvore)
    adj: Dict[int, List[int]] = {r.id: [] for r in rooms}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    leaves = [rid for rid in adj if len(adj[rid]) == 1]

    if len(leaves) < 1:
        return None

    # Uma porta por aresta da árvore (lado da sala filha `j` em cada aresta (i,j)): entra na sala só com chave.
    door_positions: List[Vec2] = []
    door_for_child: Dict[int, Vec2] = {}
    treasure_positions: List[Vec2] = []

    for i, j in edges:
        child = rooms[j]
        parent_room = rooms[i]
        bbox_neighbors: List[Vec2] = []
        for c in corridor_cells:
            cx, cy = c
            touch = False
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if child.x <= nx < child.x + child.w and child.y <= ny < child.y + child.h:
                    touch = True
                    break
            if touch:
                bbox_neighbors.append(c)
        if not bbox_neighbors:
            return None
        pc = parent_room.center()
        bbox_neighbors.sort(key=lambda q: _manhattan(q, pc))
        door_cell = bbox_neighbors[0]
        if door_cell in door_positions:
            return None
        door_positions.append(door_cell)
        door_for_child[j] = door_cell
        terrain[door_cell[1]][door_cell[0]] = Terrain.DOOR

    # Tesouro só em salas-folha; validar que não há brecha (outra abertura) além da porta dessa folha
    for leaf_id in leaves:
        leaf = rooms[leaf_id]
        door_cell = door_for_child.get(leaf_id)
        if door_cell is None:
            return None
        if not _leaf_single_door_exit(leaf, door_cell, terrain, width, height):
            return None
        tx, ty = leaf.center()
        if terrain[ty][tx] != Terrain.FLOOR:
            found = False
            for yy in range(leaf.y, leaf.y + leaf.h):
                for xx in range(leaf.x, leaf.x + leaf.w):
                    if terrain[yy][xx] == Terrain.FLOOR:
                        tx, ty = xx, yy
                        found = True
                        break
                if found:
                    break
            if not found:
                return None
        treasure_positions.append((tx, ty))

    n_doors = len(door_positions)
    n_monsters_target = max(0, int(len(corridor_cells) * monster_density))
    # Poucos monstros para manter o espaço de estados da BFS tratável (PCG gerar-e-testar)
    n_monsters_target = min(n_monsters_target, 6)
    monster_candidates = [c for c in corridor_cells if c not in door_positions]
    if len(monster_candidates) < n_monsters_target:
        n_monsters_target = len(monster_candidates)
    rng.shuffle(monster_candidates)
    monster_positions = monster_candidates[:n_monsters_target]

    # chaves: uma por porta (mínimo). Nunca dentro do interior de sala-folha
    # (ficaria inacessível atrás da porta dessa folha).
    leaf_id_set = set(leaves)

    def room_id_at(cell: Vec2) -> Optional[int]:
        cx, cy = cell
        for r in rooms:
            if r.x <= cx < r.x + r.w and r.y <= cy < r.y + r.h:
                return r.id
        return None

    key_pool = [c for c in floor_cells if terrain[c[1]][c[0]] == Terrain.FLOOR]
    key_pool = [
        c
        for c in key_pool
        if c not in door_positions
        and c not in monster_positions
        and c not in treasure_positions
        and (room_id_at(c) not in leaf_id_set)
    ]
    if len(key_pool) < n_doors:
        return None
    rng.shuffle(key_pool)
    key_positions = key_pool[:n_doors]

    # Poções: uma antes de cada monstro (incl. o 1.º) → pelo menos um monstro exige M poções
    min_potions = len(monster_positions) if monster_positions else 0
    potion_count = min_potions + rng.randint(0, max(1, n_doors))
    pot_pool = [
        c
        for c in floor_cells
        if terrain[c[1]][c[0]] == Terrain.FLOOR
        and c not in door_positions
        and c not in monster_positions
        and c not in treasure_positions
        and c not in key_positions
    ]
    if len(pot_pool) < potion_count:
        potion_count = len(pot_pool)
    rng.shuffle(pot_pool)
    potion_positions = pot_pool[:potion_count]
    if len(potion_positions) < min_potions:
        return None

    # início / saída preferem salas de grau > 1 (evita colapso quando só existem folhas, ex.: 2 salas)
    non_leaves = [rid for rid in adj if len(adj[rid]) != 1]
    if not non_leaves:
        start_room = rooms[0]
        end_room = rooms[1] if len(rooms) > 1 else rooms[0]
    else:
        start_room = rooms[rng.choice(non_leaves)]
        end_candidates = [rid for rid in non_leaves if rid != start_room.id]
        if not end_candidates:
            end_room = start_room
        else:
            end_room = rooms[rng.choice(end_candidates)]

    def pick_floor_in_room(room: Room) -> Optional[Vec2]:
        cells = [c for c in room.cells() if terrain[c[1]][c[0]] == Terrain.FLOOR]
        if not cells:
            return None
        cells.sort(key=lambda c: _manhattan(c, room.center()))
        return cells[len(cells) // 2]

    start = pick_floor_in_room(start_room)
    exitp = pick_floor_in_room(end_room)
    if start is None or exitp is None or start == exitp:
        return None

    # Preenchimento exterior (não andável): tile (2,0) da folha — só o limiar da dungeon usa parede (0,0).
    _mark_exterior_void(terrain, width, height)

    # preencher entidades
    def assign_list(positions: List[Vec2], kind: EntityKind):
        for idx, p in enumerate(positions):
            entity[p[1]][p[0]] = kind
            entity_index[p[1]][p[0]] = idx

    assign_list(monster_positions, EntityKind.MONSTER)
    assign_list(potion_positions, EntityKind.POTION)
    assign_list(key_positions, EntityKind.KEY)
    assign_list(treasure_positions, EntityKind.TREASURE)

    door_ids = {c: i for i, c in enumerate(door_positions)}
    return DungeonMap(
        width=width,
        height=height,
        terrain=terrain,
        entity=entity,
        entity_index=entity_index,
        start=start,
        exit=exitp,
        door_cells=door_positions,
        door_ids=door_ids,
        monster_cells=monster_positions,
        potion_cells=potion_positions,
        key_cells=key_positions,
        treasure_cells=treasure_positions,
        corridor_cells=corridor_cells,
        room_cells=room_cells,
    )


@dataclass(frozen=True)
class SearchState:
    x: int
    y: int
    hp: int  # 1 saudável, 0 ferido
    monsters_alive: int  # bitmask
    treasures: int  # bitmask coletados
    doors_open: int  # bitmask portas abertas
    keys_taken: int  # bitmask chaves já pegas do chão
    potions_taken: int  # bitmask poções já consumidas
    # 1 = já consumiu poção desde o último monstro (ou início); obrigatório antes de cada combate
    may_fight: int


def _key_inventory(kt: int, doors_open: int) -> int:
    """Chaves no inventário = pegas no chão − já gastas ao abrir portas."""
    return kt.bit_count() - doors_open.bit_count()


def validate_dungeon(dm: DungeonMap, max_states: int = 5_000_000) -> Tuple[bool, Optional[List[Vec2]]]:
    """BFS no produto (mapa × vida × chaves × máscaras). Retorna (ok, caminho posições)."""
    nm = len(dm.monster_cells)
    nt = len(dm.treasure_cells)
    nd = len(dm.door_cells)
    nk = len(dm.key_cells)
    np_ = len(dm.potion_cells)
    # Limites práticos de bitmask (Python int aguenta mais; o gargalo é tempo de BFS)
    if nm > 18 or nt > 18 or nd > 18 or nk > 18 or np_ > 18:
        return False, None

    d_pos = {dm.door_cells[i]: i for i in range(nd)}

    full_treasure = (1 << nt) - 1 if nt else 0
    full_doors = (1 << nd) - 1 if nd else 0
    start_alive = (1 << nm) - 1 if nm else 0
    may0 = 0 if nm > 0 else 1

    start = SearchState(dm.start[0], dm.start[1], 1, start_alive, 0, 0, 0, 0, may0)

    q: deque[SearchState] = deque([start])
    parent: Dict[SearchState, Optional[SearchState]] = {start: None}
    visited: Set[SearchState] = {start}
    goal_state: Optional[SearchState] = None

    while q and len(visited) < max_states:
        s = q.popleft()
        at_exit = s.x == dm.exit[0] and s.y == dm.exit[1]
        if at_exit and s.treasures == full_treasure and s.monsters_alive == 0 and s.doors_open == full_doors:
            goal_state = s
            break
        for nx, ny in _neigh4((s.x, s.y)):
            if not _in_bounds(dm.width, dm.height, (nx, ny)):
                continue
            ter = int(dm.terrain[ny][nx])
            ent = int(dm.entity[ny][nx])
            idx = int(dm.entity_index[ny][nx])

            hp = s.hp
            monsters = s.monsters_alive
            treasures = s.treasures
            doors = s.doors_open
            kt = s.keys_taken
            pt = s.potions_taken
            may_fight = s.may_fight

            if ter == Terrain.WALL or ter == Terrain.VOID:
                continue
            if ter == Terrain.DOOR:
                bit = d_pos.get((nx, ny))
                if bit is None:
                    continue
                if (doors >> bit) & 1:
                    pass  # porta aberta, trafega como piso
                else:
                    if _key_inventory(kt, doors) <= 0:
                        continue
                    doors |= 1 << bit
            # interações em entidades do destino
            if ent == EntityKind.MONSTER and idx >= 0:
                if (monsters >> idx) & 1 == 0:
                    pass
                else:
                    if hp == 0:
                        continue  # morte
                    if may_fight == 0:
                        continue  # falta consumir poção antes deste combate
                    hp = 0
                    monsters &= ~(1 << idx)
                    may_fight = 0
            elif ent == EntityKind.POTION and idx >= 0:
                if (pt >> idx) & 1:
                    pass  # já consumida
                else:
                    hp = 1
                    pt |= 1 << idx
                    may_fight = 1
            elif ent == EntityKind.KEY and idx >= 0:
                if (kt >> idx) & 1:
                    pass
                else:
                    kt |= 1 << idx
            elif ent == EntityKind.TREASURE and idx >= 0:
                treasures |= 1 << idx

            ns = SearchState(nx, ny, hp, monsters, treasures, doors, kt, pt, may_fight)
            if ns in visited:
                continue
            visited.add(ns)
            parent[ns] = s
            q.append(ns)

    if goal_state is None:
        return False, None

    path: List[Vec2] = []
    cur: Optional[SearchState] = goal_state
    while cur is not None:
        path.append((cur.x, cur.y))
        cur = parent[cur]
    path.reverse()
    return True, path


def generate_until_valid(
    max_tries: int = 200,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[DungeonMap, List[Vec2]]:
    rng = random.Random(seed)
    for t in range(max_tries):
        dm = build_dungeon(rng=rng, **kwargs)
        if dm is None:
            continue
        ok, path = validate_dungeon(dm)
        if ok and path:
            return dm, path
    raise RuntimeError("Não foi possível gerar mapa válido no limite de tentativas; ajuste parâmetros ou max_tries.")


# --- Renderização ---

TILE = 16


def load_sprites(sheet_path: Path) -> Dict[str, Image.Image]:
    """Recorta a folha PCG-sprites-dungeon (48×48, 9×16). VOID = 3.º tile da 1.ª linha (gx=2, gy=0)."""
    sheet = Image.open(sheet_path).convert("RGBA")
    names = [
        ("wall", 0, 0),
        ("door", 1, 0),
        ("void", 2, 0),
        ("monster", 0, 1),
        ("potion", 1, 1),
        ("key", 2, 1),
        ("hero", 0, 2),
        ("treasure", 1, 2),
        ("floor", 2, 2),
    ]
    out: Dict[str, Image.Image] = {}
    for name, gx, gy in names:
        out[name] = sheet.crop((gx * TILE, gy * TILE, (gx + 1) * TILE, (gy + 1) * TILE))
    return out


def _bright(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def render_map(
    dm: DungeonMap,
    sprites: Dict[str, Image.Image],
    path_overlay: Optional[List[Vec2]] = None,
    scale: int = 4,
) -> Image.Image:
    tw = dm.width * TILE
    th = dm.height * TILE
    base = Image.new("RGBA", (tw, th), (0, 0, 0, 255))
    floor_bright = _bright(sprites["floor"], 1.35)
    floor_dim = _bright(sprites["floor"], 0.55)
    path_set = set(path_overlay or [])

    for y in range(dm.height):
        for x in range(dm.width):
            px, py = x * TILE, y * TILE
            ter = int(dm.terrain[y][x])
            ent = int(dm.entity[y][x])
            if ter == Terrain.WALL:
                tile = sprites["wall"]
            elif ter == Terrain.VOID:
                tile = sprites["void"]
            elif ter == Terrain.DOOR:
                tile = sprites["door"]
            else:
                if (x, y) == dm.start:
                    tile = floor_bright
                elif (x, y) == dm.exit:
                    tile = floor_dim
                else:
                    tile = sprites["floor"]
            base.paste(tile, (px, py), tile)
            if ent == EntityKind.MONSTER:
                t = sprites["monster"]
                base.paste(t, (px, py), t)
            elif ent == EntityKind.POTION:
                t = sprites["potion"]
                base.paste(t, (px, py), t)
            elif ent == EntityKind.KEY:
                t = sprites["key"]
                base.paste(t, (px, py), t)
            elif ent == EntityKind.TREASURE:
                t = sprites["treasure"]
                base.paste(t, (px, py), t)
            if path_overlay and (x, y) in path_set:
                overlay = Image.new("RGBA", (TILE, TILE), (80, 200, 255, 90))
                base.alpha_composite(overlay, (px, py))

    hx, hy = dm.start
    hero = sprites["hero"]
    base.paste(hero, (hx * TILE, hy * TILE), hero)

    if scale != 1:
        base = base.resize((tw * scale, th * scale), Image.Resampling.NEAREST)
    return base


def save_approved_preview(dm: DungeonMap, path: List[Vec2], out_path: Path, sprites: Dict[str, Image.Image]) -> None:
    img = render_map(dm, sprites, path_overlay=path, scale=4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def summarize(dm: DungeonMap) -> str:
    lines = [
        f"Dimensões: {dm.width}x{dm.height}",
        f"Salas-folha (tesouros+portas): {len(dm.treasure_cells)}",
        f"Portas: {len(dm.door_cells)} | Chaves colocadas: {len(dm.key_cells)}",
        f"Monstros (só corredor): {len(dm.monster_cells)} | Poções: {len(dm.potion_cells)}",
        f"Início {dm.start} | Saída {dm.exit}",
    ]
    return "\n".join(lines)


# Mesmos argumentos que `python3 dungeon_pcg.py` e que o notebook deve usar para mapa idêntico.
# `monster_density` omitido de propósito (default 0.12 em `build_dungeon`).
DEMO_GENERATE_KWARGS: Dict[str, object] = {
    "seed": 14,
    "max_tries": 200,
    "width": 48,
    "height": 40,
    "n_rooms": 5,
}


def generate_demo_map() -> Tuple[DungeonMap, List[Vec2]]:
    """Mapa reprodutível igual ao da CLI / entrega (mesmo RNG que `DEMO_GENERATE_KWARGS`)."""
    return generate_until_valid(**DEMO_GENERATE_KWARGS)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    sprite_path = here / "assets" / "PCG-sprites-dungeon.png"
    sprites = load_sprites(sprite_path)
    dm, path = generate_demo_map()
    print(summarize(dm))
    print("Comprimento do caminho de solução:", len(path))
    img = render_map(dm, sprites, path_overlay=path)
    out = here / "assets" / "mapa_aprovado_demo.png"
    img.save(out)
    print("Salvo:", out)
