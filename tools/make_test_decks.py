"""Generate the 10 Master Duel meta-style test decks in tests/decks/.

Usage: uv run python tools/make_test_decks.py [--out DIR]

Cards are keyed by password (the card names in comments are for review
only; they were resolved once from cards.cdb, original printings only). Extra Deck monsters are routed to
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

STAPLES: list[tuple[int, int]] = [  # (password, copies); names are comments for review only
    (14558127, 3),  # Ash Blossom & Joyous Spring
    (10045474, 3),  # Infinite Impermanence
    (23434538, 2),  # Maxx "C"
    (97268402, 2),  # Effect Veiler
    (24224830, 1),  # Called by the Grave
    (65681983, 1),  # Crossout Designator
    (25311006, 2),  # Triple Tactics Talent
    (27204311, 2),  # Nibiru, the Primal Being
    (73642296, 2),  # Ghost Belle & Haunted Mansion
    (94145021, 2),  # Droll & Lock Bird
    (54693926, 2),  # Dark Ruler No More
    (24299458, 2),  # Forbidden Droplet
    (8267140, 2),  # Cosmic Cyclone
    (15693423, 2),  # Evenly Matched
]

GENERIC_EXTRA: list[int] = [
    29301450,  # S:P Little Knight
    65741786,  # I:P Masquerena
    38342335,  # Knightmare Unicorn
    2857636,  # Knightmare Phoenix
    86066372,  # Accesscode Talker
    41999284,  # Linkuriboh
    60303245,  # Salamangreat Almiraj
    98127546,  # Underworld Goddess of the Closed World
    71818935,  # Moon of the Closed Heaven
    94259633,  # Relinquished Anima
    90590303,  # Number 41: Bagooska the Terribly Tired Tapir
    84815190,  # Baronne de Fleur
    27548199,  # Borreload Savage Dragon
    90448279,  # Divine Arsenal AA-ZEUS - Sky Thunder
    46772449,  # Evilswarm Exciton Knight
]

DECKS: dict[str, list[tuple[int, int]]] = {
    "snake_eye": [
        (9674034, 3),  # Snake-Eye Ash
        (45663742, 2),  # Snake-Eye Oak
        (90241276, 3),  # Snake-Eyes Poplar
        (12058741, 1),  # Snake-Eye Birch
        (72270339, 1),  # Diabellstar the Black Witch
        (89023486, 3),  # Original Sinful Spoils - Snake-Eye
        (80845034, 1),  # WANTED: Seeker of Sinful Spoils
        (24081957, 2),  # Sinful Spoils of Subversion - Snake-Eye
        (60764609, 1),  # Fiendsmith Engraver
        (85106525, 1),  # Bonfire
        (2772337, 1),  # Promethean Princess, Bestower of Flames
        (58071334, 1),  # Snake-Eyes Doomed Dragon
        (48452496, 1),  # Snake-Eyes Flamberge Dragon
        (2463794, 1),  # Fiendsmith's Requiem
        (49867899, 1),  # Fiendsmith's Sequence
        (46640168, 1),  # Fiendsmith's Lacrima
    ],
    "yubel": [
        (78371393, 2),  # Yubel
        (90829280, 3),  # Spirit of Yubel
        (62318994, 1),  # Samsara D Lotus
        (93729896, 3),  # Nightmare Throne
        (65261141, 1),  # Nightmare Pain
        (87532344, 1),  # Eternal Favorite
        (47172959, 1),  # Yubel - The Loving Defender Forever
        (80453041, 1),  # Phantom of Yubel
        (4779091, 1),  # Yubel - Terror Incarnate
        (31764700, 1),  # Yubel - The Ultimate Nightmare
        (80312545, 1),  # Opening of the Spirit Gates
    ],
    "labrynth": [
        (2347656, 1),  # Lovely Labrynth of the Silver Castle
        (1225009, 3),  # Arianna the Labrynth Servant
        (73602965, 1),  # Arias the Labrynth Butler
        (37629703, 1),  # Labrynth Chandraglier
        (74018812, 1),  # Labrynth Stovie Torbie
        (2511, 2),  # Labrynth Cooclock
        (5380979, 3),  # Welcome Labrynth
        (92714517, 1),  # Big Welcome Labrynth
        (33407125, 1),  # Labrynth Labyrinth
        (6351147, 1),  # Transaction Rollback
        (82956214, 1),  # Dogmatika Punishment
        (30748475, 1),  # Destructive Daruma Karma Cannon
        (83326048, 1),  # Dimensional Barrier
        (53582587, 1),  # Torrential Tribute
        (94192409, 1),  # Compulsory Evacuation Device
        (80101899, 1),  # Trap Trick
        (81497285, 1),  # Lady Labrynth of the Silver Castle
        (71607202, 1),  # Muckraker From the Underworld
    ],
    "tenpai": [
        (91810826, 3),  # Tenpai Dragon Chundra
        (39931513, 3),  # Tenpai Dragon Paidra
        (65326118, 1),  # Tenpai Dragon Fadra
        (23657016, 1),  # Tenpai Dragon Genroku
        (30336082, 3),  # Sangen Summoning
        (66730191, 3),  # Sangen Kaimen
        (82570174, 1),  # Sangenpai Bident Dragion
        (18969888, 1),  # Sangenpai Transcendent Dragion
        (22850702, 1),  # Chaos Angel
        (50954680, 1),  # Crystal Wing Synchro Dragon
        (44508094, 1),  # Stardust Dragon
        (73580471, 1),  # Black Rose Dragon
    ],
    "kashtira": [
        (32909498, 3),  # Kashtira Fenrir
        (68304193, 2),  # Kashtira Unicorn
        (31149212, 3),  # Kashtira Riseheart
        (69540484, 2),  # Kashtira Birth
        (33925864, 1),  # Kashtira Big Bang
        (71832012, 3),  # Pressured Planet Wraitsoth
        (73542331, 1),  # Kashtira Shangri-Ira
        (48626373, 2),  # Kashtira Arise-Heart
        (4928565, 1),  # Tearlaments Kashtira
        (78534861, 1),  # Scareclaw Kashtira
    ],
    "branded_despia": [
        (62962630, 3),  # Aluber the Jester of Despia
        (68468459, 3),  # Fallen of Albaz
        (36577931, 1),  # Despian Tragedy
        (99456344, 1),  # Dramaturge of Despia
        (19096726, 1),  # Tri-Brigade Mercourier
        (44362883, 1),  # Branded Fusion
        (82738008, 1),  # Branded in Red
        (36637374, 2),  # Branded Opening
        (17751597, 1),  # Branded Retribution
        (18973184, 1),  # Branded Lost
        (44146295, 1),  # Mirrorjade the Iceblade Dragon
        (87746184, 1),  # Albion the Branded Dragon
        (70534340, 1),  # Lubellion the Searing Dragon
        (41373230, 1),  # Titaniklad the Ash Dragon
        (51409648, 1),  # Rindbrumm the Striking Dragon
        (11321089, 1),  # Guardian Chimera
        (18666161, 1),  # Despian Proskenion
        (53971455, 1),  # Despian Luluwalilith
        (34848821, 1),  # Brigrand the Glory Dragon
        (24915933, 1),  # Granguignol the Dusk Dragon
        (1906812, 1),  # Sprind the Irondash Dragon
    ],
    "purrely": [
        (25550531, 3),  # Purrely
        (21347668, 2),  # Purrely Sleepy Memory
        (55584558, 2),  # Purrely Delicious Memory
        (82105704, 2),  # Purrely Happy Memory
        (29599813, 2),  # Purrely Pretty Memory
        (10780049, 2),  # Purrely Sharely!?
        (56700100, 3),  # My Friend Purrely
        (20212491, 1),  # Stray Purrely Street
        (79933029, 1),  # Purrelyly
        (52645235, 1),  # Epurrely Happiness
        (62592805, 1),  # Epurrely Noir
        (98049934, 1),  # Epurrely Beauty
        (24434049, 1),  # Epurrely Plump
        (51822687, 1),  # Expurrely Happiness
        (83827392, 1),  # Expurrely Noir
    ],
    "voiceless_voice": [
        (25801745, 3),  # Lo, the Prayers of the Voiceless Voice
        (10774240, 1),  # Skull Guardian, Protector of the Voiceless Voice
        (88284599, 1),  # Sauravis, Dragon Sage of the Voiceless Voice
        (51296484, 1),  # Saffira, Dragon Queen of the Voiceless Voice
        (61773610, 1),  # Guardian of the Voiceless Voice
        (52472775, 3),  # Prayers of the Voiceless Voice
        (39114494, 1),  # Blessing of the Voiceless Voice
        (98477480, 1),  # Barrier of the Voiceless Voice
        (86310763, 1),  # Radiance of the Voiceless Voice
        (60303688, 1),  # Dogmatika Ecclesia, the Virtuous
        (1984618, 1),  # Nadir Servant
        (10804018, 1),  # Saffira, Divine Dragon of the Voiceless Voice
    ],
    "fiendsmith_ryzeal": [
        (8633261, 3),  # Ice Ryzeal
        (35844557, 2),  # Sword Ryzeal
        (72238166, 2),  # Node Ryzeal
        (84433129, 1),  # Star Ryzeal
        (6798031, 1),  # Ryzeal Cross
        (7511613, 2),  # Ryzeal Duo Drive
        (60764609, 3),  # Fiendsmith Engraver
        (99989863, 1),  # Fiendsmith in Paradise
        (98567237, 1),  # Fiendsmith's Tract
        (35552985, 1),  # Fiendsmith's Sanct
        (26434972, 1),  # Fiendsmith Kyrie
        (28803166, 1),  # Lacrima the Crimson Tears
        (34022970, 1),  # Ext Ryzeal
        (34909328, 1),  # Ryzeal Detonator
        (2463794, 1),  # Fiendsmith's Requiem
        (49867899, 1),  # Fiendsmith's Sequence
        (46640168, 1),  # Fiendsmith's Lacrima
        (82135803, 1),  # Fiendsmith's Desirae
        (32991300, 1),  # Fiendsmith's Agnumday
        (93860227, 1),  # Necroquip Princess
    ],
    "tearlaments": [
        (572850, 3),  # Tearlaments Scheiren
        (73956664, 1),  # Tearlaments Reinoheart
        (74078255, 2),  # Tearlaments Merrli
        (37961969, 2),  # Tearlaments Havnis
        (60362066, 1),  # Tearlaments Heartbeat
        (1329620, 1),  # Tearlaments Cryme
        (77103950, 1),  # Primeval Planet Perlereino
        (63542003, 2),  # Keldo the Sacred Protector
        (25926710, 1),  # Kelbek the Ancient Vanguard
        (99937011, 1),  # Mudora the Sword Oracle
        (92731385, 1),  # Tearlaments Kitkallos
        (84330567, 1),  # Tearlaments Rulkallos
        (28226490, 1),  # Tearlaments Kaleido-Heart
        (54757758, 1),  # Mudragon of the Swamp
        (20366274, 1),  # El Shaddoll Construct
    ],
}


def build(db: CardDB, spec: list[tuple[int, int]]) -> Deck:
    unknown = [pw for pw, _ in spec + STAPLES if pw not in db] + [pw for pw in GENERIC_EXTRA if pw not in db]
    if unknown:
        raise SystemExit(f"passwords not in cards.cdb: {unknown}")
    main: list[int] = []
    extra: list[int] = []
    for pw, count in spec:
        (extra if db[pw].is_extra_deck else main).extend([pw] * count)
    for pw, count in STAPLES:
        if len(main) >= 40:
            break
        room = min(count, 3 - main.count(pw), 40 - len(main))
        main.extend([pw] * room)
    for pw in GENERIC_EXTRA:
        if len(extra) >= 15:
            break
        if pw not in extra:
            extra.append(pw)
    return Deck(tuple(main), tuple(extra[:15]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "tests" / "decks")
    args = parser.parse_args()
    db = CardDB.load()
    args.out.mkdir(parents=True, exist_ok=True)
    for deck_name, spec in DECKS.items():
        deck = build(db, spec)
        check_deck(deck, cards=db, banlist=None)
        lines = [f"#created by ygorl tools/make_test_decks.py ({deck_name})", "#main"]
        lines += [f"{pw}" for pw in deck.main]
        lines += ["#extra"] + [f"{pw}" for pw in deck.extra] + ["!side"]
        (args.out / f"{deck_name}.ydk").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(DECKS)} decks to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
