"""Card lists of the Solo player (:class:`ygorl.selfplay.loop.Player`): find the wanted card in MD's list (the
client's list in memory, else the learned panel), cards hidden in lists, and learning the panel of a pick."""

from __future__ import annotations

import json
import struct
from dataclasses import replace


from ygorl.md import layout as L
from ygorl.md.memory import RawDuel
from ygorl.md.request import Candidate, _locate
from ygorl.md.selection import own_cards, scan_pairs
from ygorl.selfplay.state import (
    LIST_NEXT,
    LIST_SHOWN,
    LIST_PAGES,
    list_more,
)


class ListsMixin:
    """Find cards in MD's card lists."""

    def learn_list(self, raw: RawDuel) -> None:
        """The list candidate that left its place since the pick is the card that was picked: label the panel
        captured on its position with it (right or wrong, so the next list of these cards can be read)."""
        if self.pending_list is None:
            return
        img, req = self.pending_list
        if img is None:
            self.pending_list = None
            return
        from ygorl.md.request import _locate

        moved = {c.cid for c in req.candidates if c.uid and _locate(raw, c.uid)[1:] != (c.location, c.sequence)}
        if len(moved) == 1:
            self.pending_list = None
            (cid,) = moved
            self.panel.learn(img, cid, self.card_stats(cid))
            self.panel.save()
            self.trace.observe(panel_learned_from_list=cid)

    def hidden_in_lists(self) -> list:
        """[password, location] of list candidates MD's lists never display (learned, kept across runs): memory
        lists Ra - Immortal Phoenix among Foolish Burial's Deck monsters, and every card of MD's list was seen
        without it (four lists in runs s29 / s31)."""
        if getattr(self, "_hidden", None) is None:
            path = self.out / "md_hidden_in_lists.json"
            self._hidden = json.loads(path.read_text()) if path.exists() else []
        return self._hidden

    def learn_hidden(self, password: int, location: str) -> None:
        hidden = self.hidden_in_lists()
        if [password, location] not in hidden:
            hidden.append([password, location])
            (self.out / "md_hidden_in_lists.json").write_text(json.dumps(hidden))
        self.trace.observe(list_learned_hidden=[password, location])
        print(f"  list: MD does not display {self.cards[password].name if password in self.cards else password}"
              f" ({location}); learned", flush=True)  # fmt: skip

    def request_from_client_list(self, raw: RawDuel, req, shown: int):
        """``req`` with the candidates of the client's list (memory) when that list is the one on screen (as many
        cards); (req, still stale)."""
        client = getattr(self, "client", None)
        try:
            got = client.list_cards(raw, places=True) if client is not None else None
        except Exception:  # noqa: BLE001
            got = None
        if not got or len(got[1]) != shown or any(e.player == raw.me and not e.uid for e in got[1]):
            return req, True
        cands = []
        for e in got[1]:
            if e.player != raw.me:
                cands.append(Candidate(0, 0, e.player, "", None))  # the opponent's: a place, never read
                continue
            p, loc, seq = _locate(raw, e.uid)
            cands.append(Candidate(e.cid, e.uid, p, loc, seq))
        self.trace.observe(list_request={"from": "client list", "candidates": [[c.cid, c.location, c.sequence]
                                                                                for c in cands]})  # fmt: skip
        return replace(req, candidates=tuple(cands)), False

    def find_by_client_list(self, want: int, want_uid: int, cards: list) -> int | None:
        """The screen position of the wanted card from the client's own list (memory; its order is the screen's,
        checked on run s33's lists), ticked and confirmed by the open-card read; None when the list does not tell
        or the tick does not confirm it (then every card is ticked and read). A list longer than the screen is
        scrolled with its arrow, one card per turn, until the card is the rightmost shown."""
        client, selection = getattr(self, "client", None), getattr(self, "selection", None)
        if client is None or selection is None or selection.address is None or not cards:
            return None
        try:
            got = client.list_cards(self.game.read(), places=True)
        except Exception:  # noqa: BLE001
            return None
        if not got:
            return None
        entries, n = got[1], len(cards)
        if len(entries) < n or (len(entries) > n and n < LIST_SHOWN):
            return None  # not the list on screen
        pw = lambda cid: self.passwords.get(cid, cid)  # noqa: E731
        hits = [
            j for j, e in enumerate(entries) if e.cid and (e.uid == want_uid if want_uid else pw(e.cid) == pw(want))
        ]
        if not hits:
            return None
        j = hits[0]
        turns = max(0, j - n + 1)
        for _ in range(turns):
            self.ui.click(*LIST_NEXT, "next cards of the list (to the card the client lists)", 1.2)
        pos = j - turns
        try:
            prev = selection.memory.read(selection.address, 8)
        except Exception:  # noqa: BLE001
            prev = None
        self.ui.click(*cards[pos], f"#{pos + 1} of {n}: the card the client lists", 0.9)
        try:
            cid, uid = struct.unpack("<II", selection.memory.read(selection.address, 8))
        except Exception:  # noqa: BLE001
            cid, uid = 0, 0
        ok = pw(cid) == pw(want) and (not want_uid or uid == want_uid)
        if not ok and prev is not None and prev[:8] == struct.pack("<II", cid, uid):
            # the open-card read did not move with the tick (it held no card: a stale address): the client's
            # order is the screen's (checked since run s33), so its place stands; the stale address is dropped
            # (Ra - Sphere Mode was ticked right, then a guess sent The Winged Dragon of Ra instead)
            selection.address = None
            self.trace.observe(list_check={"want": want, "by": "client list (unconfirmed: no open-card read)",
                                           "index": j, "turns": turns})  # fmt: skip
            return pos
        self.trace.observe(list_check={"want": want, "by": "client list", "index": j, "turns": turns, "ok": ok,
                                       "changed": prev is not None and prev[:8] != struct.pack("<II", cid, uid)})  # fmt: skip
        if not ok:
            # another card of the list: aim again by its place in the client's list (a scroll click can be
            # swallowed, and the list can hold cards the screen does not show: Foolish Burial landed two short)
            k = next((i for i, e in enumerate(entries) if e.uid and e.uid == uid), None)
            again = pos + (j - k) if k is not None else -1
            self.ui.click(*cards[pos], f"untick #{pos + 1}", 0.6)
            if 0 <= again < n and again != pos:
                self.ui.click(*cards[again], f"#{again + 1} of {n}: aimed again by the client's list", 0.9)
                try:
                    cid, uid = struct.unpack("<II", selection.memory.read(selection.address, 8))
                except Exception:  # noqa: BLE001
                    cid, uid = 0, 0
                ok = pw(cid) == pw(want) and (not want_uid or uid == want_uid)
                self.trace.observe(list_check={"want": want, "by": "client list (aimed again)", "index": j, "ok": ok})
                if ok:
                    return again
                self.ui.click(*cards[again], f"untick #{again + 1}", 0.6)
            return None
        if ok:
            return pos
        return None  # (scrolled: the card the client named is on screen, where the tick-and-read search finds it)

    def find_in_list(self, req, pick: int) -> tuple[int, bool]:
        """The screen position of candidate ``pick`` in MD's card list, and whether it is ticked already.

        MD's list is not memory's: the order differs, identical copies are listed once, and a long list scrolls
        (the cards past the sixth behind a ">" arrow). So each screen card is ticked and identified: from memory
        first (ticking writes the card into the open-card object, :mod:`ygorl.md.selection`; a read counts only
        when the object changed with the click, so a stale value never names a card), else on the information
        panel, which cannot name Spells / Traps or tell monsters with equal stats apart (an audit found 6 picks
        of 7 guessed wrong). Pages are turned with the arrow until the wanted card is found. A pick that cannot be
        confirmed is recorded as unverified."""
        cards = L.list_cards(self.ui.capture())
        cands = [c.cid for c in req.candidates]
        n = len(cards)
        if not cands or not 0 <= pick < len(cands):
            return 0, False
        pw = lambda cid: self.passwords.get(cid, cid)  # noqa: E731
        want = cands[pick]
        # copies of the wanted card in different places (Mahad's Dark Magician: the hand's or the Deck's) are
        # told apart by their uid, which only memory gives
        want_uid = req.candidates[pick].uid
        copies = [c for c in req.candidates if pw(c.cid) == pw(want)]
        by_uid = len({c.location for c in copies}) > 1 and want_uid != 0
        if n <= 1 or (len({pw(c) for c in cands}) <= 1 and not by_uid):
            return 0, False  # one card, or only copies of one card: any position is the wanted card
        direct = self.find_by_client_list(want, want_uid if by_uid else 0, cards)
        if direct is not None:
            self.pending_list = (self.ui.capture(), req)
            return direct, True
        stats = {c: self.card_stats(c) for c in set(cands)}
        seen: dict = {}
        panels = {}
        selection = getattr(self, "selection", None)

        # the open-card object is not found yet (a list asked before any hand card was opened twice: Foolish
        # Burial's Deck list on turn 1 was guessed wrong twice): calibrate on the list's own ticks
        # (every card of ours: the first scan must also find the object holding the card opened before the list)
        pairs = {(c.cid, c.uid) for c in req.candidates if c.uid}
        if selection is not None and selection.address is None:
            pairs |= {(cid, uid) for uid, cid in own_cards(self.game.read()).items()}
            selection.opened(self.game.read(), pairs)  # the scan the first tick is compared against

        def read_open():
            if selection is None or selection.address is None:
                return None
            try:
                return selection.memory.read(selection.address, 8)
            except Exception:  # noqa: BLE001
                return None

        # a Deck list's ticks did not change the open-card object at all (run s27: 30 ticks, none read): then the
        # list's own objects are watched, every address holding a candidate's (card id, uid), and the one that
        # takes the ticked card's pair is followed (diagnostics in the trace either way)
        scan = {"regions": None, "last": None, "addr": None, "still": 0}
        reads: list = []

        def scan_tick():
            if selection is None or not pairs:
                return None
            if scan["addr"] is not None:
                try:
                    cid, u = struct.unpack("<II", selection.memory.read(scan["addr"], 8))
                except Exception:  # noqa: BLE001
                    return None
                return (cid, u) if (cid, u) in pairs else None
            if scan["regions"] is None:
                scan["regions"] = selection.regions()
            found = scan_pairs(selection.memory, scan["regions"], pairs)
            last, scan["last"] = scan["last"], found
            if last is None:
                return None
            changed = [a for a, p in found.items() if a in last and last[a] != p]
            # (an object keeping the card ticked before as well: that slot takes the open slot's old pair)
            olds = {last[a] for a in changed}
            changed = [a for a in changed if found[a] not in olds] or changed
            reads.append({"changed": [hex(a) for a in changed][:6]})
            if len(changed) == 1:
                scan["addr"] = changed[0]
                return found[changed[0]]
            return None

        uids_seen: set = set()
        complete = False
        for page in range(LIST_PAGES):
            # the arrow scrolls one card (run s28: 14 cards, 6 turns showed only the first 11): after a turn,
            # only the newly shown cards on the right are ticked, right to left, until one already seen (its uid)
            before_page = len(uids_seen)
            for i in range(n) if page == 0 else reversed(range(n)):
                prev = read_open()
                if scan["still"] >= 2 and scan["last"] is None and scan["addr"] is None:
                    scan_tick()  # the baseline the next tick is compared against
                self.ui.click(*cards[i], f"#{i + 1} of {n}: which card?", 0.9)
                img = self.ui.capture()
                got, uid, how = None, 0, ""
                if scan["still"] >= 2:
                    found = scan_tick()
                    if found is not None and found[0] in set(cands):
                        (got, uid), how = found, "memory (the list's object)"
                if selection is not None and prev is None:
                    found = selection.opened(self.game.read(), pairs)  # the pair that changed with this tick
                    if found is not None and found[0] in set(cands):
                        (got, uid), how = found, "memory (calibrated on the list)"
                now = read_open()
                if prev is not None and now == prev:
                    scan["still"] += 1  # the open-card object did not move with this tick
                reads.append({"prev": prev.hex() if prev else None, "now": now.hex() if now else None})
                if got is None and now is not None and prev is not None and now != prev:
                    cid, u = struct.unpack("<II", now)
                    if cid in set(cands):
                        got, uid, how = cid, u, "memory"
                if got is None and not by_uid:
                    got, how = self.panel.identify(img, list(dict.fromkeys(cands)), stats)
                if page and uid and uid in uids_seen:
                    self.ui.click(*cards[i], f"untick #{i + 1}", 0.6)
                    break  # this card and the ones left of it were looked at on an earlier page
                uids_seen.add(uid)
                seen[(page, i)], panels[(page, i)] = got, img
                if got is not None and pw(got) == pw(want) and (not by_uid or uid == want_uid):
                    self.trace.observe(list_check={"want": want, "seen": {f"{k[0]}:{k[1]}": v for k, v in
                                                                          seen.items()}, "by": how})  # fmt: skip
                    self.pending_list = (img, req)
                    return i, True
                self.ui.click(*cards[i], f"untick #{i + 1}", 0.6)
            # the arrow can read as gone before the last card (run s29: 12 of 13 seen, the 13th the wanted one):
            # while fewer cards were seen than memory lists, turn once more (a turn showing nothing new ends it)
            unseen = len(uids_seen - {0}) < len(cands) and 0 not in uids_seen
            if page and len(uids_seen) == before_page:
                complete = True
                break  # the turn showed no new card: the list's end
            if page == LIST_PAGES - 1 or (not list_more(self.ui.capture()) and not unseen):
                break  # (no page turned after the last one looked at: a guess there clicked an unseen card)
            self.ui.click(*LIST_NEXT, "next cards of the list", 1.2)
            cards = L.list_cards(self.ui.capture())
            n = len(cards)
        last = max(k[0] for k in seen)  # the page on screen now
        unknown = [k for k, got in seen.items() if got is None]
        named = {pw(got) for got in seen.values() if got is not None}
        here = [k for k in unknown if k[0] == last]
        if len(unknown) == 1 and here and {pw(c) for c in cands} - named == {pw(want)}:
            k = unknown[0]  # every other distinct card was named: the one nobody could name is it
            self.trace.observe(list_check={"want": want, "by": "elimination"})
            self.pending_list = (panels[k], req)
            return k[1], False
        if complete and 0 not in uids_seen and want_uid not in uids_seen:
            # every card MD displays was named from memory and the wanted one is not among them: MD does not offer
            # it. Learned (the next lists leave it to the policy's other choices); this pick falls to a rule
            self.learn_hidden(pw(want), req.candidates[pick].location)
            observer = getattr(self, "observer", None)
            if observer is not None:
                observer.shadow = None  # it took the card MD does not show: out of step from here
            self.trace.rule("list-wanted-card-not-displayed", want=want)
        k = here[0] if here else (last, 0)
        self.trace.observe(list_check={"want": want, "unverified": True, "seen": {f"{k[0]}:{k[1]}": v for k, v in
                                       seen.items()}, "reads": reads[-12:]})  # fmt: skip
        print(f"  list: {self.name(want)} not identified; position {k[1] + 1} is a guess", flush=True)
        self.pending_list = (panels.get(k), req)
        return k[1], False
