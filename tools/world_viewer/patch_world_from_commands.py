#!/usr/bin/env python3
import argparse
import csv
import math
import re
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from nbt.nbt import (
    NBTFile,
    TAG_Byte,
    TAG_Compound,
    TAG_Double,
    TAG_Float,
    TAG_Int,
    TAG_List,
    TAG_Long_Array,
    TAG_String,
)
from nbt.region import RegionFile

from multisteve.mc_creative import place_tree_commands


PLACEMENT_DEFAULTS = {
    'x': 4,
    'y': 0,
    'z': 7,
    'tree_type': 'oak',
    'relative': True,
    'villager': True,
    'villager_profession': 'unemployed',
    'villager_type': 'plains',
}

SIDE_DEFAULT_COORDS = {
    'left': (4, 0, 7),
    'right': (-4, 0, 7),
}


@dataclass
class SpawnRef:
    x: float
    y: float
    z: float


@dataclass
class SectionWork:
    section_tag: TAG_Compound
    palette_names: List[str]
    indices: List[int]


class WorldPatcher:
    def __init__(self, world_dir: Path):
        self.world_dir = world_dir
        self.region_dir = world_dir / 'region'
        self.region_cache: Dict[Tuple[int, int], RegionFile] = {}
        self.chunk_cache: Dict[Tuple[int, int], Tuple[RegionFile, int, int, NBTFile]] = {}
        self.section_cache: Dict[Tuple[int, int, int], SectionWork] = {}
        self.modified_chunks: set[Tuple[int, int]] = set()

    @staticmethod
    def _chunk_coords(x: int, z: int) -> Tuple[int, int]:
        return math.floor(x / 16), math.floor(z / 16)

    @staticmethod
    def _region_coords(cx: int, cz: int) -> Tuple[int, int]:
        return math.floor(cx / 32), math.floor(cz / 32)

    @staticmethod
    def _local_chunk_in_region(cx: int, cz: int, rx: int, rz: int) -> Tuple[int, int]:
        return cx - rx * 32, cz - rz * 32

    def _get_region(self, rx: int, rz: int) -> RegionFile:
        key = (rx, rz)
        if key in self.region_cache:
            return self.region_cache[key]
        path = self.region_dir / f'r.{rx}.{rz}.mca'
        if not path.exists():
            raise FileNotFoundError(f'Missing region file: {path}')
        reg = RegionFile(str(path))
        self.region_cache[key] = reg
        return reg

    def _get_chunk(self, cx: int, cz: int) -> Tuple[RegionFile, int, int, NBTFile]:
        key = (cx, cz)
        if key in self.chunk_cache:
            return self.chunk_cache[key]

        rx, rz = self._region_coords(cx, cz)
        lx, lz = self._local_chunk_in_region(cx, cz, rx, rz)
        region = self._get_region(rx, rz)
        nbt_file = region.get_nbt(lx, lz)
        self.chunk_cache[key] = (region, lx, lz, nbt_file)
        return self.chunk_cache[key]

    @staticmethod
    def _get_tag(compound: TAG_Compound, name: str):
        for t in compound.tags:
            if t.name == name:
                return t
        return None

    @staticmethod
    def _set_or_replace_tag(compound: TAG_Compound, new_tag):
        for i, t in enumerate(compound.tags):
            if t.name == new_tag.name:
                compound.tags[i] = new_tag
                return
        compound.tags.append(new_tag)

    @staticmethod
    def _bits_for_palette(palette_len: int) -> int:
        return max(4, (palette_len - 1).bit_length())

    @staticmethod
    def _decode_blockstates(long_values: List[int], bits: int) -> List[int]:
        mask = (1 << bits) - 1
        u64 = [(v + (1 << 64)) % (1 << 64) for v in long_values]
        out = [0] * 4096
        for i in range(4096):
            bit_idx = i * bits
            li = bit_idx // 64
            start = bit_idx % 64
            if li >= len(u64):
                out[i] = 0
                continue
            val = (u64[li] >> start) & mask
            spill = start + bits - 64
            if spill > 0 and (li + 1) < len(u64):
                val |= (u64[li + 1] & ((1 << spill) - 1)) << (bits - spill)
            out[i] = int(val)
        return out

    @staticmethod
    def _encode_blockstates(indices: List[int], bits: int) -> List[int]:
        longs_len = math.ceil(4096 * bits / 64)
        u64 = [0] * longs_len
        mask = (1 << bits) - 1
        for i, idx in enumerate(indices):
            v = idx & mask
            bit_idx = i * bits
            li = bit_idx // 64
            start = bit_idx % 64
            u64[li] |= (v << start) & ((1 << 64) - 1)
            spill = start + bits - 64
            if spill > 0 and (li + 1) < longs_len:
                u64[li + 1] |= v >> (bits - spill)

        out: List[int] = []
        for v in u64:
            if v >= (1 << 63):
                out.append(v - (1 << 64))
            else:
                out.append(v)
        return out

    def _get_level_tag(self, nbt_file: NBTFile) -> TAG_Compound:
        for t in nbt_file.tags:
            if t.name == 'Level':
                return t
        raise ValueError('Chunk NBT missing Level tag')

    def _get_or_create_section_work(self, cx: int, cz: int, sec_y: int) -> SectionWork:
        key = (cx, cz, sec_y)
        if key in self.section_cache:
            return self.section_cache[key]

        _, _, _, nbt_file = self._get_chunk(cx, cz)
        level = self._get_level_tag(nbt_file)
        sections = self._get_tag(level, 'Sections')
        if sections is None:
            sections = TAG_List(name='Sections', type=TAG_Compound)
            level.tags.append(sections)

        sec_tag = None
        for s in sections.tags:
            y_tag = self._get_tag(s, 'Y')
            if y_tag is not None and int(y_tag.value) == sec_y:
                sec_tag = s
                break

        if sec_tag is None:
            sec_tag = TAG_Compound(name='')
            sec_tag.tags.append(TAG_Byte(name='Y', value=sec_y))
            palette = TAG_List(name='Palette', type=TAG_Compound)
            entry = TAG_Compound(name='')
            entry.tags.append(TAG_String(name='Name', value='minecraft:air'))
            palette.tags.append(entry)
            sec_tag.tags.append(palette)
            bs = TAG_Long_Array(name='BlockStates')
            bs.value = [0] * 256
            sec_tag.tags.append(bs)
            sections.tags.append(sec_tag)

        palette_tag = self._get_tag(sec_tag, 'Palette')
        if palette_tag is None:
            palette_tag = TAG_List(name='Palette', type=TAG_Compound)
            entry = TAG_Compound(name='')
            entry.tags.append(TAG_String(name='Name', value='minecraft:air'))
            palette_tag.tags.append(entry)
            self._set_or_replace_tag(sec_tag, palette_tag)

        blockstates_tag = self._get_tag(sec_tag, 'BlockStates')
        if blockstates_tag is None:
            bits = self._bits_for_palette(len(palette_tag.tags) or 1)
            blockstates_tag = TAG_Long_Array(name='BlockStates')
            blockstates_tag.value = [0] * math.ceil(4096 * bits / 64)
            self._set_or_replace_tag(sec_tag, blockstates_tag)

        palette_names: List[str] = []
        for p in palette_tag.tags:
            name_tag = self._get_tag(p, 'Name')
            palette_names.append(str(name_tag.value) if name_tag is not None else 'minecraft:air')

        bits = self._bits_for_palette(max(1, len(palette_names)))
        indices = self._decode_blockstates(list(blockstates_tag.value), bits)

        work = SectionWork(section_tag=sec_tag, palette_names=palette_names, indices=indices)
        self.section_cache[key] = work
        return work

    def set_block(self, x: int, y: int, z: int, block_name: str):
        block_name = normalize_block_name(block_name)
        cx, cz = self._chunk_coords(x, z)
        _, _, _, _ = self._get_chunk(cx, cz)

        sec_y = math.floor(y / 16)
        ly = y - sec_y * 16
        lx = x & 15
        lz = z & 15

        work = self._get_or_create_section_work(cx, cz, sec_y)
        if block_name not in work.palette_names:
            work.palette_names.append(block_name)
        pal_idx = work.palette_names.index(block_name)

        idx = (ly * 16 + lz) * 16 + lx
        work.indices[idx] = pal_idx
        self.modified_chunks.add((cx, cz))

    def add_villager(self, x: int, y: int, z: int, profession: str, villager_type: str):
        cx, cz = self._chunk_coords(x, z)
        _, _, _, nbt_file = self._get_chunk(cx, cz)
        level = self._get_level_tag(nbt_file)
        ents = self._get_tag(level, 'Entities')
        if ents is None:
            ents = TAG_List(name='Entities', type=TAG_Compound)
            level.tags.append(ents)
        elif getattr(ents, 'tagID', None) == 0:
            # Empty TAG_List in older chunks may serialize with element type 0 until fixed.
            ents.tagID = TAG_Compound.id

        e = TAG_Compound(name='')
        e.tags.append(TAG_String(name='id', value='minecraft:villager'))

        pos = TAG_List(name='Pos', type=TAG_Double)
        pos.tags.append(TAG_Double(value=float(x) + 0.5))
        pos.tags.append(TAG_Double(value=float(y)))
        pos.tags.append(TAG_Double(value=float(z) + 0.5))
        e.tags.append(pos)

        motion = TAG_List(name='Motion', type=TAG_Double)
        motion.tags.append(TAG_Double(value=0.0))
        motion.tags.append(TAG_Double(value=0.0))
        motion.tags.append(TAG_Double(value=0.0))
        e.tags.append(motion)

        rot = TAG_List(name='Rotation', type=TAG_Float)
        rot.tags.append(TAG_Float(value=0.0))
        rot.tags.append(TAG_Float(value=0.0))
        e.tags.append(rot)

        villager_data = TAG_Compound(name='VillagerData')
        villager_data.tags.append(TAG_String(name='profession', value=f'minecraft:{profession}'))
        villager_data.tags.append(TAG_String(name='type', value=f'minecraft:{villager_type}'))
        villager_data.tags.append(TAG_Int(name='level', value=1))
        e.tags.append(villager_data)

        e.tags.append(TAG_Byte(name='OnGround', value=1))
        e.tags.append(TAG_Byte(name='PersistenceRequired', value=1))
        e.tags.append(TAG_Int(name='Air', value=300))
        e.tags.append(TAG_Int(name='Fire', value=-1))

        ents.tags.append(e)
        self.modified_chunks.add((cx, cz))

    def _commit_section_state(self):
        for (cx, cz, _sec_y), work in self.section_cache.items():
            palette_list = TAG_List(name='Palette', type=TAG_Compound)
            for name in work.palette_names:
                c = TAG_Compound(name='')
                c.tags.append(TAG_String(name='Name', value=name))
                palette_list.tags.append(c)

            bits = self._bits_for_palette(max(1, len(work.palette_names)))
            encoded = self._encode_blockstates(work.indices, bits)
            blockstates = TAG_Long_Array(name='BlockStates')
            blockstates.value = encoded

            self._set_or_replace_tag(work.section_tag, palette_list)
            self._set_or_replace_tag(work.section_tag, blockstates)
            self.modified_chunks.add((cx, cz))

    def save(self):
        self._commit_section_state()
        for cx, cz in sorted(self.modified_chunks):
            region, lx, lz, nbt_file = self._get_chunk(cx, cz)
            region.write_chunk(lx, lz, nbt_file)

        for reg in self.region_cache.values():
            reg.close()


def parse_bool(value: str) -> bool:
    val = value.strip().lower()
    if val in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if val in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise ValueError(f'Invalid bool: {value}')


def parse_placement_spec(spec: str) -> Dict[str, object]:
    out = dict(PLACEMENT_DEFAULTS)
    raw: Dict[str, str] = {}
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        k, v = chunk.split('=', 1)
        raw[k.strip().lower()] = v.strip()

    if 'side' in raw:
        side = raw['side'].lower()
        if side not in SIDE_DEFAULT_COORDS:
            raise ValueError(f'Unknown side: {raw["side"]}')
        sx, sy, sz = SIDE_DEFAULT_COORDS[side]
        out['x'], out['y'], out['z'] = sx, sy, sz

    if 'x' in raw:
        out['x'] = int(raw['x'])
    if 'y' in raw:
        out['y'] = int(raw['y'])
    if 'z' in raw:
        out['z'] = int(raw['z'])
    if 'tree_type' in raw:
        out['tree_type'] = raw['tree_type'].lower()
    if 'relative' in raw:
        out['relative'] = parse_bool(raw['relative'])
    if 'villager' in raw:
        out['villager'] = parse_bool(raw['villager'])
    if 'villager_profession' in raw:
        out['villager_profession'] = raw['villager_profession']
    if 'villager_type' in raw:
        out['villager_type'] = raw['villager_type']
    return out


def parse_command_log_for_placements(command_log: Path) -> List[str]:
    text = command_log.read_text(encoding='utf-8').strip()
    argv = shlex.split(text)
    specs: List[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == '--placement' and i + 1 < len(argv):
            specs.append(argv[i + 1])
            i += 2
            continue
        i += 1
    return specs


def normalize_block_name(name: str) -> str:
    n = name.strip()
    if n.startswith('minecraft:'):
        return n
    return f'minecraft:{n}'


def parse_coord(token: str, base: float) -> int:
    t = token.strip()
    if t.startswith('~'):
        delta = 0.0
        if len(t) > 1:
            delta = float(t[1:])
        return math.floor(base + delta)
    return math.floor(float(t))


def parse_summon_villager_data(nbt_blob: str) -> Tuple[str, str]:
    prof = 'unemployed'
    typ = 'plains'
    m_prof = re.search(r'profession:([a-z_]+)', nbt_blob)
    m_type = re.search(r'type:([a-z_]+)', nbt_blob)
    if m_prof:
        prof = m_prof.group(1)
    if m_type:
        typ = m_type.group(1)
    return prof, typ


def load_spawn(path: Path) -> SpawnRef:
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        first = next(reader)
    return SpawnRef(float(first['xpos']), float(first['ypos']), float(first['zpos']))


def apply_chat_command(patcher: WorldPatcher, cmd: str, spawn: SpawnRef):
    line = cmd.strip()
    if not line:
        return
    if line.startswith('/'):
        line = line[1:]
    parts = shlex.split(line)
    if not parts:
        return

    op = parts[0].lower()
    if op == 'fill' and len(parts) >= 8:
        x1 = parse_coord(parts[1], spawn.x)
        y1 = parse_coord(parts[2], spawn.y)
        z1 = parse_coord(parts[3], spawn.z)
        x2 = parse_coord(parts[4], spawn.x)
        y2 = parse_coord(parts[5], spawn.y)
        z2 = parse_coord(parts[6], spawn.z)
        block = normalize_block_name(parts[7])

        for x in range(min(x1, x2), max(x1, x2) + 1):
            for y in range(min(y1, y2), max(y1, y2) + 1):
                for z in range(min(z1, z2), max(z1, z2) + 1):
                    patcher.set_block(x, y, z, block)
        return

    if op == 'setblock' and len(parts) >= 5:
        x = parse_coord(parts[1], spawn.x)
        y = parse_coord(parts[2], spawn.y)
        z = parse_coord(parts[3], spawn.z)
        block = normalize_block_name(parts[4])
        patcher.set_block(x, y, z, block)
        return

    if op == 'summon' and len(parts) >= 5 and parts[1].lower() == 'villager':
        x = parse_coord(parts[2], spawn.x)
        y = parse_coord(parts[3], spawn.y)
        z = parse_coord(parts[4], spawn.z)
        prof, typ = parse_summon_villager_data(parts[5] if len(parts) > 5 else '')
        patcher.add_villager(x, y, z, profession=prof, villager_type=typ)
        return


def build_tree_commands_from_placements(placement_specs: List[str]) -> List[str]:
    out: List[str] = []
    for spec in placement_specs:
        p = parse_placement_spec(spec)
        out += place_tree_commands(
            int(p['x']),
            int(p['y']),
            int(p['z']),
            tree_type=str(p['tree_type']),
            relative=bool(p['relative']),
            villager=bool(p['villager']),
            villager_profession=str(p['villager_profession']),
            villager_type=str(p['villager_type']),
        )
    return out


def verify_patched_world(world: Path, spawn: SpawnRef) -> Tuple[bool, Dict[str, object]]:
    from anvil import Region, Chunk

    sx = math.floor(spawn.x)
    sy = math.floor(spawn.y)
    sz = math.floor(spawn.z)

    checks = {
        'oak_top': (sx + 4, sy + 4, sz + 7, {'oak_log'}),
        'birch_top': (sx - 4, sy + 4, sz + 7, {'birch_log'}),
    }
    details: Dict[str, object] = {}

    def get_block_id(x: int, y: int, z: int) -> str:
        cx, cz = x // 16, z // 16
        rx, rz = cx // 32, cz // 32
        rf = world / 'region' / f'r.{rx}.{rz}.mca'
        reg = Region.from_file(str(rf))
        ch = Chunk.from_region(reg, cx, cz)
        b = ch.get_block(x & 15, y, z & 15)
        return str(getattr(b, 'id', getattr(b, 'name', b))).split(':')[-1]

    ok = True
    for name, (x, y, z, allowed) in checks.items():
        bid = get_block_id(x, y, z)
        details[name] = {'coord': (x, y, z), 'block': bid, 'allowed': sorted(list(allowed))}
        if bid not in allowed:
            ok = False

    # Verify villager in the spawn chunk neighborhood.
    villager_count = 0
    rx_min = (sx - 24) // 512
    rx_max = (sx + 24) // 512
    rz_min = (sz - 24) // 512
    rz_max = (sz + 24) // 512
    for rx in range(rx_min, rx_max + 1):
        for rz in range(rz_min, rz_max + 1):
            rf = world / 'region' / f'r.{rx}.{rz}.mca'
            if not rf.exists():
                continue
            reg = RegionFile(str(rf))
            for lc in reg.get_chunk_coords():
                lx, lz = lc['x'], lc['z']
                nbt_file = reg.get_nbt(lx, lz)
                level = None
                for t in nbt_file.tags:
                    if t.name == 'Level':
                        level = t
                        break
                if level is None:
                    continue
                ents = None
                for t in level.tags:
                    if t.name == 'Entities':
                        ents = t
                        break
                if ents is None:
                    continue
                for e in ents.tags:
                    eid = None
                    pos = None
                    for et in e.tags:
                        if et.name == 'id':
                            eid = str(et.value)
                        elif et.name == 'Pos':
                            pos = et
                    if eid != 'minecraft:villager' or pos is None or len(pos.tags) < 3:
                        continue
                    ex = float(pos.tags[0].value)
                    ey = float(pos.tags[1].value)
                    ez = float(pos.tags[2].value)
                    if abs(ex - (sx + 4)) <= 2 and abs(ez - (sz + 7)) <= 2 and abs(ey - sy) <= 6:
                        villager_count += 1
            reg.close()

    details['villager_count_near_left_tree'] = villager_count
    if villager_count < 1:
        ok = False

    return ok, details


def main():
    p = argparse.ArgumentParser(description='Patch copied Minecraft world files from placement command specs.')
    p.add_argument('--world-src', required=True)
    p.add_argument('--world-dst', required=True)
    p.add_argument('--location-stats', required=True)
    p.add_argument('--command-log', required=True)
    p.add_argument('--report', required=True)
    args = p.parse_args()

    src = Path(args.world_src)
    dst = Path(args.world_dst)
    report = Path(args.report)

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    spawn = load_spawn(Path(args.location_stats))
    specs = parse_command_log_for_placements(Path(args.command_log))
    commands = build_tree_commands_from_placements(specs)

    patcher = WorldPatcher(dst)
    for cmd in commands:
        apply_chat_command(patcher, cmd, spawn)
    patcher.save()

    ok, details = verify_patched_world(dst, spawn)

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        'status=' + ('ok' if ok else 'failed') + '\n'
        + f'spawn=({spawn.x},{spawn.y},{spawn.z})\n'
        + f'placement_specs={specs}\n'
        + f'verify_details={details}\n',
        encoding='utf-8',
    )

    if not ok:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
