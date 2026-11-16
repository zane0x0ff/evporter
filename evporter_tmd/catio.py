import struct

U = lambda d, o: struct.unpack_from('<I', d, o)[0] if o + 4 <= len(d) else None


def _pad(n, a):
    return (n + a - 1) // a * a if a else n


class Leaf:
    def __init__(self, data):
        self.data = bytes(data)

    def __repr__(self):
        return 'Leaf(%d)' % len(self.data)


class Pack:
    """a texture pack: DDS files in name-list order"""

    def __init__(self, files):
        self.files = [bytes(f) for f in files]

    def bytes(self):
        n = len(self.files)
        hs = 12 + 4 * n
        offs, o = [], 0
        for f in self.files:
            offs.append(o)
            o += len(f)
        return struct.pack('<3I%dI' % n, hs, n, hs + o, *offs) + b''.join(self.files)

    def __repr__(self):
        return 'Pack(%d)' % len(self.files)


class Wrap:
    def __init__(self, type_, child, align=0x100, raw=None, extra=b''):
        self.type, self.child, self.align, self.raw, self.extra = type_, child, align, raw, extra

    def __repr__(self):
        return 'Wrap(%d, %r)' % (self.type, self.child)


class List:
    def __init__(self, ctype, children, align=0x100):
        self.ctype, self.children, self.align = ctype, children, align

    def names(self):
        c = self.children[0] if self.children else None
        if isinstance(c, Leaf):
            return names_of(c.data)
        return []

    def __repr__(self):
        return 'List(%d, %r)' % (self.ctype, self.children)


def names_of(b):
    out = []
    for line in b.split(b'\n'):
        line = line.strip(b'\r\0')
        if line.endswith(b','):
            line = line[:-1]
        if line:
            out.append(line.decode('latin-1'))
    return out


def names_bytes(names):
    return b''.join(n.encode('latin-1') + b',\r\n' for n in names)


def _is_pack(d):
    hs, n = U(d, 0), U(d, 4)
    if hs is None or n is None or not 0 < n < 0x1000 or hs != 12 + 4 * n or len(d) < hs + 4 or d[hs:hs + 4] != b'DDS ':
        return False
    tot = U(d, 8)
    offs = struct.unpack_from('<%dI' % n, d, 12)
    return tot == len(d) and list(offs) == sorted(offs) and offs[0] == 0 and offs[-1] < tot - hs


def _node(d, o, end, depth):
    """the node at o (a wrap / list) or None if it is not one"""
    if depth > 24:
        return None
    k, n, c, a = U(d, o), U(d, o + 4), U(d, o + 8), U(d, o + 12)
    if None in (k, n, c, a) or a not in (0x40, 0x100):
        return None
    if k == 1 and n == 1 and c == 0:
        size, typ, cnt = U(d, o + 16), U(d, o + 20), U(d, o + 24)
        if size is None or o + a + size > end:
            return None
        child = _node(d, o + a, o + a + size, depth + 1)
        if child is None:
            return None
        hdr = d[o:o + a]
        extra = hdr[28:]
        return Wrap(typ, child, a, extra=extra if extra.strip(b'\0') else b'')
    if k == 0 and 0 < n < 0x10000 and U(d, o + 16) == 0:
        hl = 20 + 8 * n
        if o + hl > end:
            return None
        offs = struct.unpack_from('<%dI' % n, d, o + 20)
        sizes = struct.unpack_from('<%dI' % n, d, o + 20 + 4 * n)
        kids = []
        for off, sz in zip(offs, sizes):
            if off < hl or o + off + sz > end or off % a:
                return None
            sub = d[o + off:o + off + sz]
            ch = _node(d, o + off, o + off + sz, depth + 1)
            if ch is None:
                ch = Pack.__new__(Pack) if _is_pack(sub) else None
                if ch is not None:
                    hs, cnt = U(sub, 0), U(sub, 4)
                    po = struct.unpack_from('<%dI' % cnt, sub, 12) + (len(sub) - hs,)
                    ch.files = [sub[hs + po[i]:hs + po[i + 1]] for i in range(cnt)]
                    if ch.bytes() != sub:
                        ch = None
                if ch is None:
                    ch = Leaf(sub)
            kids.append(ch)
        return List(c, kids, a)
    return None


def read(d):
    """catalogue bytes -> (root node, trailing bytes after it)"""
    d = bytes(d)
    root = _node(d, 0, len(d), 0)
    if root is None:
        raise ValueError('not a catalogue (no wrap / list node at offset 0)')
    return root, d[len(_write(root)):]


def _write(node):
    if isinstance(node, Leaf):
        return node.data
    if isinstance(node, Pack):
        return node.bytes()
    if isinstance(node, Wrap):
        body = _write(node.child)
        body += b'\0' * (_pad(len(body), node.align) - len(body))
        cnt = len(node.child.children) if isinstance(node.child, List) else 1
        hdr = struct.pack('<7I', 1, 1, 0, node.align, len(body), node.type, cnt) + node.extra
        return hdr + b'\0' * (node.align - len(hdr)) + body
    if isinstance(node, List):
        a = node.align
        n = len(node.children)
        hl = _pad(20 + 8 * n, a)
        parts, offs, sizes, o = [], [], [], hl
        for ch in node.children:
            b = _write(ch)
            offs.append(o)
            sizes.append(len(b))
            pb = b + b'\0' * (_pad(len(b), a) - len(b))
            parts.append(pb)
            o += len(pb)
        hdr = struct.pack('<5I', 0, n, node.ctype, a, 0) + struct.pack('<%dI' % n, *offs) + struct.pack('<%dI' % n, *sizes)
        return hdr + b'\0' * (hl - len(hdr)) + b''.join(parts)
    raise TypeError(node)


def write(root, trailing=b''):
    b = _write(root)
    if isinstance(root, Wrap):
        pass
    return b + trailing


def walk(node, path=()):
    """yield (path, node, parent list, index) for every node below"""
    if isinstance(node, Wrap):
        yield from walk(node.child, path + (('wrap', node.type),))
    elif isinstance(node, List):
        nm = node.names()
        first = 1 if nm and isinstance(node.children[0], Leaf) else 0
        for i, ch in enumerate(node.children):
            label = nm[i - first] if first and 0 <= i - first < len(nm) else None
            yield path, ch, node, i, label
            if isinstance(ch, (Wrap, List)):
                yield from walk(ch, path + ((label or i),))


def leaves(root):
    """{name: (list, index)} of the named leaves (model, tables) and {name: (pack, index)} of the textures"""
    named, textures = {}, {}
    for path, ch, parent, i, label in walk(root):
        if isinstance(ch, Pack):
            nm = parent.names()
            for k in range(len(ch.files)):
                textures.setdefault(nm[k] if k < len(nm) else 'tex%02d' % k, (ch, k))
        elif isinstance(ch, Leaf) and label is not None:
            named.setdefault(label, (parent, i))
    return named, textures
