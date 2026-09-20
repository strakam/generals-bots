"""Decode a generals.io ``.gior`` replay (the format the site serves from its public S3 buckets,
``https://generalsio-replays-{na,eu,bot}.s3.amazonaws.com/<id>.gior``) into the row format that
``replay_agreement.prepare`` consumes. The client runs ``JSON.parse(LZString.decompressFromUint8Array(bytes))``
and unpacks a positional array; both are mirrored here. Decoder vendored from agent_jax/online/scrape_replay.py."""
import json

# LZString.decompressFromUint8Array — vendored port of the canonical algorithm.
# The bundle stores the compressed stream as big-endian 16-bit words; decode
# those back to a UTF-16 string, then run the standard LZString decompressor.
# --------------------------------------------------------------------------- #
def _lz_decompress(length, reset_value, get_next):
    dictionary = ["", "", ""]            # indices 0,1,2 are placeholders, never read
    enlarge_in = 4
    dict_size = 4
    num_bits = 3
    result = []
    data_val = get_next(0)
    data_position = reset_value
    data_index = 1

    def read_bits(nbits):
        nonlocal data_val, data_position, data_index
        bits = 0
        power = 1
        maxpower = 1 << nbits
        while power != maxpower:
            resb = data_val & data_position
            data_position >>= 1
            if data_position == 0:
                data_position = reset_value
                data_val = get_next(data_index)
                data_index += 1
            if resb > 0:
                bits |= power
            power <<= 1
        return bits

    first = read_bits(2)
    if first == 2:
        return ""
    c = chr(read_bits(8 if first == 0 else 16))
    dictionary.append(c)                 # index 3
    w = c
    result.append(c)

    while True:
        if data_index > length:
            return ""
        cbits = read_bits(num_bits)
        if cbits == 0:
            dictionary.append(chr(read_bits(8)))
            cbits = dict_size
            dict_size += 1
            enlarge_in -= 1
        elif cbits == 1:
            dictionary.append(chr(read_bits(16)))
            cbits = dict_size
            dict_size += 1
            enlarge_in -= 1
        elif cbits == 2:
            return "".join(result)

        if enlarge_in == 0:
            enlarge_in = 1 << num_bits
            num_bits += 1

        if cbits < len(dictionary):
            entry = dictionary[cbits]
        elif cbits == dict_size:
            entry = w + w[0]
        else:
            return None
        result.append(entry)

        dictionary.append(w + entry[0])
        dict_size += 1
        enlarge_in -= 1
        w = entry
        if enlarge_in == 0:
            enlarge_in = 1 << num_bits
            num_bits += 1


def decompress_from_uint8array(data: bytes) -> str:
    n = len(data) // 2
    s = "".join(chr(256 * data[2 * i] + data[2 * i + 1]) for i in range(n))
    out = _lz_decompress(len(s), 32768, lambda i: ord(s[i]) if i < len(s) else 0)
    if out is None:
        raise ValueError("LZString decompression failed (corrupt .gior?)")
    # Recombine any UTF-16 surrogate pairs (e.g. emoji in usernames).
    try:
        out = out.encode("utf-16-le", "surrogatepass").decode("utf-16-le")
    except Exception:
        pass
    return out


def deserialize(data: bytes) -> dict:
    """Field order mirrors ``Replay.deserialize`` in the generals.io client bundle (format 16+ adds
    general trades; earlier fields are unchanged)."""
    arr = json.loads(decompress_from_uint8array(data))
    it = iter(arr)
    nxt = lambda: next(it)                                                          # noqa: E731
    nxt_or = lambda d: (lambda v: d if v is None else v)(next(it, None))            # noqa: E731
    r = {}
    r["version"], r["id"], r["mapWidth"], r["mapHeight"] = nxt(), nxt(), nxt(), nxt()
    r["usernames"], r["stars"], r["cities"], r["cityArmies"], r["generals"] = nxt(), nxt(), nxt(), nxt(), nxt()
    r["mountains"] = nxt_or([])
    r["moves"] = [list(m) for m in nxt()]                                            # [player, start, end, is50, turn]
    r["afks"] = [{"index": a[0], "turn": a[1]} for a in nxt()]
    r["teams"], r["map"] = nxt(), nxt()
    r["neutrals"], r["neutralArmies"], r["swamps"], r["chat"] = nxt_or([]), nxt_or([]), nxt_or([]), nxt_or([])
    r["playerColors"], r["lights"] = nxt_or(None), nxt_or([])
    settings = nxt_or([1, 0.5, 0.5, 0]); r["speed"] = settings[0] if len(settings) > 0 else 1
    r["modifiers"], r["observatories"], r["lookouts"], r["deserts"] = nxt_or([]), nxt_or([]), nxt_or([]), nxt_or([])
    r["player_transforms"], r["pings"], r["generalTrades"] = nxt_or(None), nxt_or([]), nxt_or([])
    r["tunnels"], r["tunnelLimits"] = nxt_or([]), nxt_or([])                        # tunnels admit at most `limit` armies per move
    r["chessClockTimingsByMove"] = nxt_or([])
    r["strongholds"], r["strongholdStrengths"] = [], []
    if r["version"] >= 18:
        _density, _smin, _smax = nxt_or(None), nxt_or(None), nxt_or(None)
        r["strongholds"], r["strongholdStrengths"] = nxt_or([]), nxt_or([])
    return r


def row(path: str) -> dict:
    """The dict ``replay_agreement.prepare`` expects, plus ``afks`` and ``extras`` (non-standard map features)."""
    r = deserialize(open(path, "rb").read())
    extras = {k: r[k] for k in ("swamps", "deserts", "lights", "observatories", "lookouts", "neutrals", "modifiers", "tunnels", "strongholds") if r.get(k)}
    return dict(id=r["id"], version=r["version"], mapWidth=r["mapWidth"], mapHeight=r["mapHeight"], usernames=r["usernames"],
                cities=r["cities"], cityArmies=r["cityArmies"], generals=r["generals"], mountains=r["mountains"],
                moves=r["moves"], afks=r["afks"], teams=r["teams"], extras=extras,
                lookouts=r["lookouts"], observatories=r["observatories"], tunnels=r["tunnels"], tunnelLimits=r["tunnelLimits"],
                strongholds=r["strongholds"], generalTrades=r["generalTrades"])
