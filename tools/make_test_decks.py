"""Generate the 10 Master Duel meta-style test decks in tests/decks/.

Usage: uv run python tools/make_test_decks.py [--out DIR]

Decks are written as card *names* and resolved to passwords through
cards.cdb, so no password is typed by hand. Extra Deck monsters are routed to
the Extra Deck by card type, the Main Deck is padded to 40 from a staples list
and the Extra Deck is capped at 15. The lists approximate archetype cores seen
in the Master Duel meta (2024-2026); they are engine test fixtures, not
tournament-accurate lists, and are not checked against the MD banlist (which
has no upstream source yet, see T5.1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ygorl.cards.cdb import CardDB
from ygorl.cards.legality import check_deck
from ygorl.cards.ydk import Deck

ROOT = Path(__file__).resolve().parents[1]

STAPLES = [
    ("Ash Blossom & Joyous Spring", 3),
    ("Infinite Impermanence", 3),
    ("Maxx \"C\"", 2),
    ("Effect Veiler", 2),
    ("Called by the Grave", 1),
    ("Crossout Designator", 1),
    ("Triple Tactics Talent", 2),
    ("Nibiru, the Primal Being", 2),
    ("Ghost Belle & Haunted Mansion", 2),
    ("Droll & Lock Bird", 2),
    ("Dark Ruler No More", 2),
    ("Forbidden Droplet", 2),
    ("Cosmic Cyclone", 2),
    ("Evenly Matched", 2),
]

GENERIC_EXTRA = [
    "S:P Little Knight", "I:P Masquerena", "Knightmare Unicorn", "Knightmare Phoenix", "Accesscode Talker",
    "Linkuriboh", "Salamangreat Almiraj", "Underworld Goddess of the Closed World", "Moon of the Closed Heaven",
    "Relinquished Anima", "Number 41: Bagooska the Terribly Tired Tapir", "Baronne de Fleur",
    "Borreload Savage Dragon", "Divine Arsenal AA-ZEUS - Sky Thunder", "Evilswarm Exciton Knight",
]  # fmt: skip

DECKS: dict[str, list[tuple[str, int]]] = {
    "snake_eye": [
        ("Snake-Eye Ash", 3), ("Snake-Eye Oak", 2), ("Snake-Eyes Poplar", 3), ("Snake-Eye Birch", 1),
        ("Diabellstar the Black Witch", 1), ("Original Sinful Spoils - Snake-Eye", 3),
        ("WANTED: Seeker of Sinful Spoils", 1), ("Sinful Spoils of Subversion - Snake-Eye", 2),
        ("Fiendsmith Engraver", 1), ("Bonfire", 1),
        ("Promethean Princess, Bestower of Flames", 1), ("Snake-Eyes Doomed Dragon", 1),
        ("Snake-Eyes Flamberge Dragon", 1), ("Fiendsmith's Requiem", 1), ("Fiendsmith's Sequence", 1),
        ("Fiendsmith's Lacrima", 1),
    ],
    "yubel": [
        ("Yubel", 2), ("Spirit of Yubel", 3), ("Samsara D Lotus", 1), ("Nightmare Throne", 3),
        ("Nightmare Pain", 1), ("Eternal Favorite", 1), ("Yubel - The Loving Defender Forever", 1),
        ("Phantom of Yubel", 1), ("Yubel - Terror Incarnate", 1), ("Yubel - The Ultimate Nightmare", 1),
        ("Opening of the Spirit Gates", 1),
    ],
    "labrynth": [
        ("Lovely Labrynth of the Silver Castle", 1), ("Arianna the Labrynth Servant", 3),
        ("Arias the Labrynth Butler", 1), ("Labrynth Chandraglier", 1), ("Labrynth Stovie Torbie", 1),
        ("Labrynth Cooclock", 2), ("Welcome Labrynth", 3), ("Big Welcome Labrynth", 1),
        ("Labrynth Labyrinth", 1), ("Transaction Rollback", 1), ("Dogmatika Punishment", 1),
        ("Destructive Daruma Karma Cannon", 1), ("Dimensional Barrier", 1), ("Torrential Tribute", 1),
        ("Compulsory Evacuation Device", 1), ("Trap Trick", 1),
        ("Lady Labrynth of the Silver Castle", 1), ("Muckraker From the Underworld", 1),
    ],
    "tenpai": [
        ("Tenpai Dragon Chundra", 3), ("Tenpai Dragon Paidra", 3), ("Tenpai Dragon Fadra", 1),
        ("Tenpai Dragon Genroku", 1), ("Sangen Summoning", 3), ("Sangen Kaimen", 3),
        ("Sangenpai Bident Dragion", 1), ("Sangenpai Transcendent Dragion", 1), ("Chaos Angel", 1),
        ("Crystal Wing Synchro Dragon", 1), ("Stardust Dragon", 1), ("Black Rose Dragon", 1),
    ],
    "kashtira": [
        ("Kashtira Fenrir", 3), ("Kashtira Unicorn", 2), ("Kashtira Riseheart", 3), ("Kashtira Birth", 2),
        ("Kashtira Big Bang", 1), ("Pressured Planet Wraitsoth", 3), ("Kashtira Shangri-Ira", 1),
        ("Kashtira Arise-Heart", 2), ("Tearlaments Kashtira", 1), ("Scareclaw Kashtira", 1),
    ],
    "branded_despia": [
        ("Aluber the Jester of Despia", 3), ("Fallen of Albaz", 3), ("Despian Tragedy", 1),
        ("Dramaturge of Despia", 1), ("Tri-Brigade Mercourier", 1), ("Branded Fusion", 1),
        ("Branded in Red", 1), ("Branded Opening", 2), ("Branded Retribution", 1), ("Branded Lost", 1),
        ("Mirrorjade the Iceblade Dragon", 1), ("Albion the Branded Dragon", 1),
        ("Lubellion the Searing Dragon", 1), ("Titaniklad the Ash Dragon", 1),
        ("Rindbrumm the Striking Dragon", 1), ("Guardian Chimera", 1), ("Despian Proskenion", 1),
        ("Despian Luluwalilith", 1), ("Brigrand the Glory Dragon", 1), ("Granguignol the Dusk Dragon", 1),
        ("Sprind the Irondash Dragon", 1),
    ],
    "purrely": [
        ("Purrely", 3), ("Purrely Sleepy Memory", 2), ("Purrely Delicious Memory", 2),
        ("Purrely Happy Memory", 2), ("Purrely Pretty Memory", 2), ("Purrely Sharely!?", 2),
        ("My Friend Purrely", 3), ("Stray Purrely Street", 1), ("Purrelyly", 1),
        ("Epurrely Happiness", 1), ("Epurrely Noir", 1), ("Epurrely Beauty", 1), ("Epurrely Plump", 1),
        ("Expurrely Happiness", 1), ("Expurrely Noir", 1),
    ],
    "voiceless_voice": [
        ("Lo, the Prayers of the Voiceless Voice", 3), ("Skull Guardian, Protector of the Voiceless Voice", 1),
        ("Sauravis, Dragon Sage of the Voiceless Voice", 1), ("Saffira, Dragon Queen of the Voiceless Voice", 1),
        ("Guardian of the Voiceless Voice", 1), ("Prayers of the Voiceless Voice", 3),
        ("Blessing of the Voiceless Voice", 1), ("Barrier of the Voiceless Voice", 1),
        ("Radiance of the Voiceless Voice", 1), ("Dogmatika Ecclesia, the Virtuous", 1),
        ("Nadir Servant", 1), ("Saffira, Divine Dragon of the Voiceless Voice", 1),
    ],
    "fiendsmith_ryzeal": [
        ("Ice Ryzeal", 3), ("Sword Ryzeal", 2), ("Node Ryzeal", 2), ("Star Ryzeal", 1), ("Ryzeal Cross", 1),
        ("Ryzeal Duo Drive", 2), ("Fiendsmith Engraver", 3), ("Fiendsmith in Paradise", 1),
        ("Fiendsmith's Tract", 1), ("Fiendsmith's Sanct", 1), ("Fiendsmith Kyrie", 1),
        ("Lacrima the Crimson Tears", 1), ("Ext Ryzeal", 1), ("Ryzeal Detonator", 1),
        ("Fiendsmith's Requiem", 1), ("Fiendsmith's Sequence", 1), ("Fiendsmith's Lacrima", 1),
        ("Fiendsmith's Desirae", 1), ("Fiendsmith's Agnumday", 1), ("Necroquip Princess", 1),
    ],
    "tearlaments": [
        ("Tearlaments Scheiren", 3), ("Tearlaments Reinoheart", 1), ("Tearlaments Merrli", 2),
        ("Tearlaments Havnis", 2), ("Tearlaments Heartbeat", 1), ("Tearlaments Cryme", 1),
        ("Primeval Planet Perlereino", 1), ("Keldo the Sacred Protector", 2), ("Kelbek the Ancient Vanguard", 1),
        ("Mudora the Sword Oracle", 1), ("Tearlaments Kitkallos", 1), ("Tearlaments Rulkallos", 1),
        ("Tearlaments Kaleido-Heart", 1), ("Mudragon of the Swamp", 1), ("El Shaddoll Construct", 1),
    ],
}


def by_name(db: CardDB) -> dict[str, int]:
    names: dict[str, int] = {}
    for pw in sorted(db):
        card = db[pw]
        if card.alias or not card.ot & 0x3:  # originals only, OCG/TCG cards only
            continue
        names.setdefault(card.name, pw)
    return names


def build(db: CardDB, names: dict[str, int], spec: list[tuple[str, int]]) -> Deck:
    main: list[int] = []
    extra: list[int] = []
    missing = [n for n, _ in spec if n not in names]
    if missing:
        raise SystemExit(f"unknown card names: {missing}")
    for name, count in spec:
        pw = names[name]
        (extra if db[pw].is_extra_deck else main).extend([pw] * count)
    for name, count in STAPLES:
        if len(main) >= 40:
            break
        pw = names[name]
        room = min(count, 3 - main.count(pw), 40 - len(main))
        main.extend([pw] * room)
    for name in GENERIC_EXTRA:
        if len(extra) >= 15:
            break
        pw = names[name]
        if pw not in extra:
            extra.append(pw)
    return Deck(tuple(main), tuple(extra[:15]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "tests" / "decks")
    args = parser.parse_args()
    db = CardDB.load()
    names = by_name(db)
    args.out.mkdir(parents=True, exist_ok=True)
    for deck_name, spec in DECKS.items():
        deck = build(db, names, spec)
        check_deck(deck, cards=db, banlist=None)
        lines = [f"#created by ygorl tools/make_test_decks.py ({deck_name})", "#main"]
        lines += [f"{pw}" for pw in deck.main]
        lines += ["#extra"] + [f"{pw}" for pw in deck.extra] + ["!side"]
        (args.out / f"{deck_name}.ydk").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(DECKS)} decks to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
