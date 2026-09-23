# md-2026-09 构建与校对报告

由 `ygorl env build` 生成，重新构建时覆盖。流程见 [docs/data.md](../../../docs/data.md)。

## 禁限表校对

状态：**已校对**。对照游戏内「禁止・限制卡一览」逐条核对下表；不一致的卡写进 `review/banlist-overrides.lflist.conf`（`<password> <张数> --原因`，张数 3 表示解除），然后 `ygorl env build md-2026-09 --offline --reviewed-by <名字>` 重新生成。

最近一次 MD 禁限更新：2026-08-31，[Master Duel: Forbidden / Limited List Update](https://www.masterduelmeta.com/news/august-30-2026/master-duel-forbidden-list-update/)。

与独立来源（tools/crosscheck_banlist.py）的交叉核对记录：[banlist-crosscheck.md](banlist-crosscheck.md)。

共 207 张：禁止 108、限制 73、准限制 26；人工修正 0 条。

| 密码 | 卡名 | MD | TCG | OCG | 备注 |
|---|---|---|---|---|---|
| 21044178 | Abyss Dweller | 禁止 | 禁止 | 禁止 |  |
| 62320425 | Agido the Ancient Sentinel | 禁止 | 禁止 | 禁止 |  |
| 91869203 | Amazoness Archer | 禁止 | 无限制 | 禁止 |  |
| 58921041 | Anti-Spell Fragrance | 禁止 | 限制 | 限制 |  |
| 4280258 | Apollousa, Bow of the Goddess | 禁止 | 禁止 | 禁止 | 异画 4280259 合并 |
| 43262273 | Appointer of the Red Lotus | 禁止 | 禁止 | 无限制 |  |
| 59509952 | Archlord Kristya | 禁止 | 禁止 | 无限制 |  |
| 80237445 | Artifact Mjollnir | 禁止 | 禁止 | 无限制 |  |
| 20292186 | Artifact Scythe | 禁止 | 禁止 | 禁止 |  |
| 19740112 | Barrier Statue of the Drought | 禁止 | 禁止 | 无限制 |  |
| 73356503 | Barrier Statue of the Stormwinds | 禁止 | 禁止 | 禁止 |  |
| 27552504 | Beatrice, Lady of the Eternal | 禁止 | 禁止 | 禁止 |  |
| 9929398 | Blackwing - Gofu the Vague Shadow | 禁止 | 禁止 | 禁止 |  |
| 1041278 | Branded Expulsion | 禁止 | 禁止 | 禁止 |  |
| 69243953 | Butterfly Dagger - Elma | 禁止 | 禁止 | 禁止 |  |
| 24224830 | Called by the Grave | 禁止 | 限制 | 禁止 |  |
| 11384280 | Cannon Soldier | 禁止 | 无限制 | 禁止 |  |
| 14702066 | Cannon Soldier MK-2 | 禁止 | 无限制 | 禁止 |  |
| 59750328 | Card of Demise | 禁止 | 限制 | 无限制 |  |
| 57953380 | Card of Safe Return | 禁止 | 禁止 | 禁止 |  |
| 95727991 | Catapult Turtle | 禁止 | 无限制 | 禁止 |  |
| 3040496 | Chaos Ruler, the Chaotic Magical Dragon | 禁止 | 禁止 | 禁止 |  |
| 60682203 | Cold Wave | 禁止 | 禁止 | 禁止 |  |
| 17375316 | Confiscation | 禁止 | 禁止 | 禁止 |  |
| 50588353 | Crystron Halqifibrax | 禁止 | 禁止 | 禁止 |  |
| 69015963 | Cyber-Stein | 禁止 | 禁止 | 禁止 |  |
| 15341821 | Dandylion | 禁止 | 禁止 | 禁止 |  |
| 44763025 | Delinquent Duo | 禁止 | 禁止 | 禁止 |  |
| 23557835 | Dimension Fusion | 禁止 | 禁止 | 禁止 |  |
| 31423101 | Divine Sword - Phoenix Blade | 禁止 | 无限制 | 禁止 |  |
| 8903700 | Djinn Releaser of Rituals | 禁止 | 禁止 | 禁止 |  |
| 17412721 | Elder Entity Norden | 禁止 | 禁止 | 限制 |  |
| 38273745 | Evilswarm Ouroboros | 禁止 | 禁止 | 禁止 |  |
| 78706415 | Fiber Jar | 禁止 | 禁止 | 禁止 |  |
| 93369354 | Fishborg Blaster | 禁止 | 禁止 | 禁止 |  |
| 42009836 | Fossil Dyna Pachycephalo | 禁止 | 禁止 | 无限制 |  |
| 42703248 | Giant Trunade | 禁止 | 禁止 | 禁止 |  |
| 55204071 | Gimmick Puppet Nightmare | 禁止 | 禁止 | 无限制 |  |
| 79571449 | Graceful Charity | 禁止 | 禁止 | 禁止 |  |
| 75732622 | Grinder Golem | 禁止 | 禁止 | 禁止 |  |
| 59537380 | Guardragon Agarpain | 禁止 | 禁止 | 禁止 |  |
| 86148577 | Guardragon Elpy | 禁止 | 禁止 | 禁止 |  |
| 87639778 | Harpie's Feather Storm | 禁止 | 禁止 | 禁止 |  |
| 62242678 | Hot Red Dragon Archfiend King Calamity | 禁止 | 禁止 | 禁止 |  |
| 35984222 | Ido the Supreme Magical Force | 禁止 | 无限制 | 无限制 |  |
| 61740673 | Imperial Order | 禁止 | 禁止 | 禁止 |  |
| 41855169 | Jowgen the Spiritualist | 禁止 | 禁止 | 无限制 |  |
| 35059553 | Kaiser Colosseum | 禁止 | 禁止 | 禁止 |  |
| 25926710 | Kelbek the Ancient Vanguard | 禁止 | 禁止 | 禁止 |  |
| 10158145 | Knightmare Corruptor Iblee | 禁止 | 无限制 | 无限制 |  |
| 28566710 | Last Turn | 禁止 | 禁止 | 禁止 |  |
| 85602018 | Last Will | 禁止 | 禁止 | 禁止 |  |
| 34086406 | Lavalval Chain | 禁止 | 禁止 | 禁止 |  |
| 57421866 | Level Eater | 禁止 | 禁止 | 禁止 |  |
| 17178486 | Life Equalizer | 禁止 | 无限制 | 禁止 |  |
| 85243784 | Linkross | 禁止 | 禁止 | 禁止 |  |
| 76815942 | Lyrilusc - Independent Nightingale | 禁止 | 无限制 | 无限制 |  |
| 32723153 | Magical Explosion | 禁止 | 限制 | 禁止 |  |
| 34206604 | Magical Scientist | 禁止 | 禁止 | 禁止 |  |
| 68059897 | Maliss <Q> Red Ransom | 禁止 | 无限制 | 禁止 | 不在卡池 |
| 34906152 | Mass Driver | 禁止 | 禁止 | 禁止 |  |
| 96782886 | Mind Master | 禁止 | 禁止 | 限制 |  |
| 41482598 | Mirage of Nightmare | 禁止 | 无限制 | 禁止 |  |
| 76375976 | Mystic Mine | 禁止 | 禁止 | 禁止 |  |
| 61049315 | Naturia Rosewhip | 禁止 | 禁止 | 禁止 |  |
| 54719828 | Number 16: Shock Master | 禁止 | 禁止 | 禁止 |  |
| 90590303 | Number 41: Bagooska the Terribly Tired Tapir | 禁止 | 无限制 | 禁止 | 异画 90590304 合并 |
| 35772782 | Number 67: Pair-a-Dice Smasher | 禁止 | 禁止 | 禁止 |  |
| 63504681 | Number 86: Heroic Champion - Rhongomyniad | 禁止 | 禁止 | 禁止 |  |
| 95474755 | Number 89: Diablosis the Mind Hacker | 禁止 | 禁止 | 无限制 |  |
| 58820923 | Number 95: Galaxy-Eyes Dark Matter Dragon | 禁止 | 禁止 | 禁止 |  |
| 52653092 | Number S0: Utopic ZEXAL | 禁止 | 禁止 | 禁止 |  |
| 72537897 | Obedience Schooled | 禁止 | 限制 | 禁止 |  |
| 34945480 | Outer Entity Azathot | 禁止 | 禁止 | 禁止 |  |
| 74191942 | Painful Choice | 禁止 | 禁止 | 禁止 |  |
| 19619755 | Performapal Five-Rainbow Magician | 禁止 | 无限制 | 无限制 |  |
| 90884403 | Phantasmal Lord Ultimitl Bishbaalkin | 禁止 | 禁止 | 无限制 |  |
| 23558733 | Phoenixian Cluster Amaryllis | 禁止 | 禁止 | 禁止 |  |
| 55144522 | Pot of Greed | 禁止 | 禁止 | 禁止 |  |
| 84211599 | Pot of Prosperity | 禁止 | 限制 | 限制 |  |
| 70828912 | Premature Burial | 禁止 | 无限制 | 禁止 |  |
| 77103950 | Primeval Planet Perlereino | 禁止 | 无限制 | 限制 |  |
| 27174286 | Return from the Different Dimension | 禁止 | 禁止 | 禁止 |  |
| 93016201 | Royal Oppression | 禁止 | 禁止 | 禁止 |  |
| 6798031 | Ryzeal Cross | 禁止 | 无限制 | 无限制 |  |
| 68462976 | Secret Village of the Spellcasters | 禁止 | 无限制 | 限制 |  |
| 57585212 | Self-Destruct Button | 禁止 | 禁止 | 无限制 |  |
| 73468603 | Set Rotation | 禁止 | 限制 | 限制 |  |
| 3280747 | Sixth Sense | 禁止 | 禁止 | 禁止 |  |
| 63789924 | Smoke Grenade of the Thief | 禁止 | 禁止 | 禁止 |  |
| 54447022 | Soul Charge | 禁止 | 禁止 | 禁止 |  |
| 20663556 | Substitoad | 禁止 | 限制 | 禁止 |  |
| 23516703 | Summon Limit | 禁止 | 禁止 | 禁止 |  |
| 77679716 | Superheavy Samurai Soulbreaker Armor | 禁止 | 无限制 | 禁止 |  |
| 74078255 | Tearlaments Merrli | 禁止 | 限制 | 限制 |  |
| 63101919 | Tempest Magician | 禁止 | 禁止 | 禁止 |  |
| 73628505 | Terraforming | 禁止 | 限制 | 限制 |  |
| 42829885 | The Forceful Sentry | 禁止 | 禁止 | 禁止 |  |
| 88071625 | The Tyrant Neptune | 禁止 | 禁止 | 禁止 |  |
| 90809975 | Toadally Awesome | 禁止 | 无限制 | 无限制 |  |
| 79875176 | Toon Cannon Soldier | 禁止 | 无限制 | 禁止 |  |
| 22593417 | Topologic Gumblar Dragon | 禁止 | 禁止 | 禁止 |  |
| 64697231 | Trap Dustshoot | 禁止 | 禁止 | 禁止 |  |
| 88581108 | True King of All Calamities | 禁止 | 禁止 | 禁止 |  |
| 80604091 | Ultimate Offering | 禁止 | 禁止 | 禁止 | 异画 80604092 合并 |
| 83152482 | Union Carrier | 禁止 | 禁止 | 禁止 |  |
| 5851097 | Vanity's Emptiness | 禁止 | 禁止 | 禁止 |  |
| 16923472 | Wind-Up Hunter | 禁止 | 禁止 | 禁止 |  |
| 1475311 | Allure of Darkness | 限制 | 无限制 | 限制 |  |
| 32181268 | Amano-Iwato | 限制 | 无限制 | 无限制 |  |
| 6602300 | Blaze Fenix, the Burning Bombardment Bird | 限制 | 无限制 | 限制 |  |
| 85106525 | Bonfire | 限制 | 限制 | 限制 |  |
| 44362883 | Branded Fusion | 限制 | 限制 | 限制 |  |
| 33854624 | Bystial Magnamhut | 限制 | 限制 | 限制 |  |
| 72892473 | Card Destruction | 限制 | 限制 | 限制 |  |
| 65681983 | Crossout Designator | 限制 | 限制 | 限制 |  |
| 31425736 | Cupsy☆Yummy | 限制 | 限制 | 无限制 |  |
| 91800273 | Dimension Shifter | 限制 | 禁止 | 限制 |  |
| 75003700 | Dracotail Lukias | 限制 | 准限制 | 无限制 |  |
| 33396948 | Exodia the Forbidden One | 限制 | 限制 | 限制 |  |
| 34022970 | Ext Ryzeal | 限制 | 限制 | 限制 |  |
| 55623480 | Fairy Tail - Snow | 限制 | 限制 | 无限制 |  |
| 28126717 | Floowandereeze and the Magnificent Map | 限制 | 无限制 | 无限制 |  |
| 81439173 | Foolish Burial | 限制 | 限制 | 限制 | 异画 81439174 合并 |
| 35726888 | Foolish Burial Goods | 限制 | 无限制 | 准限制 |  |
| 75500286 | Gold Sarcophagus | 限制 | 限制 | 限制 |  |
| 53334471 | Gozen Match | 限制 | 限制 | 限制 |  |
| 18144506 | Harpie's Feather Duster | 限制 | 限制 | 限制 | 异画 18144507 合并 |
| 19613556 | Heavy Storm | 限制 | 禁止 | 限制 |  |
| 17266660 | Herald of Orange Light | 限制 | 无限制 | 限制 |  |
| 79606837 | Herald of the Arc Light | 限制 | 禁止 | 限制 |  |
| 15397015 | Inspector Boarder | 限制 | 无限制 | 无限制 |  |
| 1845204 | Instant Fusion | 限制 | 限制 | 限制 |  |
| 63542003 | Keldo the Sacred Protector | 限制 | 限制 | 限制 |  |
| 17209452 | Kewl Tune Rotary | 限制 | 禁止 | 限制 |  |
| 7902349 | Left Arm of the Forbidden One | 限制 | 限制 | 限制 |  |
| 44519536 | Left Leg of the Forbidden One | 限制 | 限制 | 限制 |  |
| 96676583 | Maliss <P> Chessy Cat | 限制 | 无限制 | 无限制 |  |
| 32061192 | Maliss <P> Dormouse | 限制 | 限制 | 限制 |  |
| 69272449 | Maliss <P> White Rabbit | 限制 | 限制 | 无限制 |  |
| 68337209 | Maliss in Underground | 限制 | 准限制 | 无限制 |  |
| 23434538 | Maxx "C" | 限制 | 禁止 | 限制 |  |
| 45171524 | Mitsurugi Prayers | 限制 | 无限制 | 限制 |  |
| 83764718 | Monster Reborn | 限制 | 限制 | 限制 | 异画 83764719 合并 |
| 99937011 | Mudora the Sword Oracle | 限制 | 限制 | 限制 |  |
| 33782437 | One Day of Peace | 限制 | 限制 | 限制 |  |
| 2295440 | One for One | 限制 | 限制 | 限制 |  |
| 38814750 | PSY-Framegear Gamma | 限制 | 限制 | 限制 |  |
| 74586817 | PSY-Framelord Omega | 限制 | 禁止 | 限制 |  |
| 98645731 | Pot of Duality | 限制 | 无限制 | 无限制 |  |
| 49238328 | Pot of Extravagance | 限制 | 无限制 | 准限制 |  |
| 70369116 | Predaplant Verte Anaconda | 限制 | 禁止 | 禁止 |  |
| 71832012 | Pressured Planet Wraitsoth | 限制 | 无限制 | 无限制 |  |
| 32548318 | Rahu Dracotail | 限制 | 限制 | 限制 |  |
| 32807846 | Reinforcement of the Army | 限制 | 限制 | 限制 |  |
| 70903634 | Right Arm of the Forbidden One | 限制 | 限制 | 限制 |  |
| 8124921 | Right Leg of the Forbidden One | 限制 | 限制 | 限制 |  |
| 90846359 | Rivalry of Warlords | 限制 | 限制 | 限制 |  |
| 94445733 | Runick Destruction | 限制 | 无限制 | 无限制 |  |
| 92107604 | Runick Fountain | 限制 | 无限制 | 限制 |  |
| 66730191 | Sangen Kaimen | 限制 | 限制 | 限制 |  |
| 30336082 | Sangen Summoning | 限制 | 限制 | 限制 |  |
| 82732705 | Skill Drain | 限制 | 限制 | 限制 |  |
| 52340444 | Sky Striker Mecha - Hornet Drones | 限制 | 限制 | 限制 |  |
| 9674034 | Snake-Eye Ash | 限制 | 无限制 | 限制 |  |
| 41420027 | Solemn Judgment | 限制 | 限制 | 限制 |  |
| 40605147 | Solemn Strike | 限制 | 无限制 | 无限制 |  |
| 84749824 | Solemn Warning | 限制 | 无限制 | 准限制 |  |
| 54562327 | Stake your Soul! | 限制 | 无限制 | 无限制 |  |
| 90361010 | Superheavy Samurai Soulpiercer | 限制 | 无限制 | 限制 |  |
| 35844557 | Sword Ryzeal | 限制 | 准限制 | 限制 |  |
| 99243014 | Synchro Overtake | 限制 | 限制 | 限制 |  |
| 60306277 | Synchro Zone | 限制 | 无限制 | 无限制 |  |
| 4928565 | Tearlaments Kashtira | 限制 | 无限制 | 限制 |  |
| 92731385 | Tearlaments Kitkallos | 限制 | 禁止 | 禁止 |  |
| 74920585 | Tearlaments Sulliek | 限制 | 无限制 | 无限制 |  |
| 91810826 | Tenpai Dragon Chundra | 限制 | 限制 | 限制 |  |
| 11110587 | That Grass Looks Greener | 限制 | 限制 | 限制 |  |
| 24207889 | There Can Be Only One | 限制 | 限制 | 限制 |  |
| 80845034 | WANTED: Seeker of Sinful Spoils | 限制 | 无限制 | 限制 |  |
| 85115440 | Zoodiac Broadbull | 限制 | 禁止 | 禁止 |  |
| 13332685 | Ame no Habakiri no Mitsurugi | 准限制 | 限制 | 无限制 |  |
| 8628798 | D.D. Dynamite | 准限制 | 无限制 | 无限制 |  |
| 7375867 | Dracotail Mululu | 准限制 | 限制 | 限制 |  |
| 94145021 | Droll & Lock Bird | 准限制 | 准限制 | 限制 | 异画 94145022 合并 |
| 60764609 | Fiendsmith Engraver | 准限制 | 无限制 | 限制 |  |
| 28642461 | K9-66a Jokul | 准限制 | 限制 | 限制 |  |
| 68304193 | Kashtira Unicorn | 准限制 | 无限制 | 限制 |  |
| 6153210 | Ketu Dracotail | 准限制 | 限制 | 限制 |  |
| 16387555 | Kewl Tune Cue | 准限制 | 无限制 | 无限制 |  |
| 16509007 | Kewl Tune Mix | 准限制 | 无限制 | 无限制 |  |
| 89392810 | Kewl Tune Reco | 准限制 | 无限制 | 无限制 |  |
| 78058681 | Kewl Tune Synchro | 准限制 | 无限制 | 准限制 |  |
| 10966439 | Marshmao☆Yummy | 准限制 | 无限制 | 限制 |  |
| 28297833 | Necroface | 准限制 | 无限制 | 无限制 |  |
| 67115133 | Radiant Typhoon Chant | 准限制 | 限制 | 无限制 |  |
| 20508881 | Radiant Typhoon Vision | 准限制 | 无限制 | 无限制 |  |
| 23002292 | Red Reboot | 准限制 | 禁止 | 限制 |  |
| 30430448 | Runick Freezing Curses | 准限制 | 无限制 | 无限制 |  |
| 67835547 | Runick Slumber | 准限制 | 无限制 | 无限制 |  |
| 31562086 | Runick Tip | 准限制 | 无限制 | 无限制 |  |
| 48130397 | Super Polymerization | 准限制 | 无限制 | 准限制 |  |
| 572850 | Tearlaments Scheiren | 准限制 | 限制 | 限制 |  |
| 69299029 | Treasures of the Kings | 准限制 | 无限制 | 无限制 |  |
| 9091064 | Vanquish Soul Jiaolong | 准限制 | 无限制 | 无限制 |  |
| 29302858 | Vanquish Soul Razen | 准限制 | 无限制 | 限制 |  |
| 78872731 | Zoodiac Ratpier | 准限制 | 限制 | 准限制 |  |

## 卡池

YGOPRODECK `format=master duel` 列出 13865 张，跳过衍生物 7 张，卡池 13858 个密码（异画归并到原卡）。

按卡名对应（YGOPRODECK 的 id 不在 BabelCDB）：

- Barrel Dragon：81480461 → 81480460
- Mercurium the Living Quicksilver：101303084 → 22984000

## Meta 卡组

窗口 2026-09-04 起的 masterduelmeta 卡表 848 份（计入份额的 560 份，其余是 statsWeight 为 0 的活动 / 特殊赛事卡表），145 个卡组类型；份额 ≥ 1.0% 且有合法卡表的前 20 个类型入选，共 20 套，份额合计 71.7%。

| 卡组 | 份额 | 卡表数 | 合法卡表 | 入选 | 不合法原因（卡表数） |
|---|---|---|---|---|---|
| Dracotail | 8.9% | 53 | 53 | 是 |  |
| Clown Crew | 8.2% | 68 | 68 | 是 |  |
| Sky Striker | 7.9% | 44 | 44 | 是 |  |
| Branded | 6.1% | 38 | 38 | 是 |  |
| Resonators | 5.4% | 44 | 44 | 是 |  |
| Kewl Tune | 4.5% | 28 | 28 | 是 |  |
| Elfnote Kewl Tune | 3.8% | 23 | 23 | 是 |  |
| Magistus Fairy Tail | 3.4% | 25 | 25 | 是 |  |
| Yummy | 2.9% | 18 | 18 | 是 |  |
| Lunalight | 2.7% | 19 | 19 | 是 |  |
| Elfnote | 2.1% | 25 | 25 | 是 |  |
| Vanquish Soul K9 | 2.1% | 13 | 13 | 是 |  |
| Radiant Typhoon Zoodiac | 2.0% | 15 | 15 | 是 |  |
| Ryzeal Mitsurugi | 2.0% | 11 | 11 | 是 |  |
| Maliss | 1.8% | 10 | 10 | 是 |  |
| Tearlaments | 1.8% | 14 | 14 | 是 |  |
| Blue-Eyes | 1.6% | 15 | 15 | 是 |  |
| HEROs | 1.6% | 10 | 10 | 是 |  |
| Odion | 1.6% | 9 | 9 | 是 |  |
| Orcust | 1.6% | 12 | 12 | 是 |  |
| Synchrons | 1.6% | 10 | 10 |  |  |
| R.B. | 1.2% | 11 | 11 |  |  |
| Radiant Typhoon | 1.2% | 7 | 7 |  |  |
| Cyber Dragon | 1.1% | 8 | 8 |  |  |
| Gem-Knights | 1.1% | 8 | 8 |  |  |
| Tellarknight | 1.1% | 6 | 6 |  |  |
| Tenpai Dragon | 1.1% | 6 | 6 |  |  |
| Magnet Warrior | 0.9% | 7 | 7 |  |  |
| Mitsurugi | 0.9% | 6 | 6 |  |  |
| Snake-Eye Yummy | 0.9% | 5 | 5 |  |  |
| Azamina | 0.7% | 5 | 5 |  |  |
| D/D/D | 0.7% | 6 | 6 |  |  |
| Railway | 0.7% | 6 | 6 |  |  |
| Solfachord Yummy | 0.7% | 4 | 4 |  |  |
| Artmage | 0.5% | 9 | 9 |  |  |
| Darklord | 0.5% | 14 | 14 |  |  |
| Enneacraft | 0.5% | 4 | 4 |  |  |
| GMX | 0.5% | 3 | 3 |  |  |
| Labrynth | 0.5% | 3 | 3 |  |  |
| Magical Musket | 0.5% | 6 | 6 |  |  |
| Mermail Atlantean | 0.5% | 3 | 3 |  |  |
| Predaplant | 0.5% | 6 | 6 |  |  |

按日期统计的卡表数与在新禁限表下不合法的卡表数。新禁限表生效后仍有不合法卡表，说明禁限表可能有误（需校对）；不合法卡表只出现在窗口开头，说明禁限表在那之后才生效，应以 `--since` 推迟窗口起点：

| 日期 | 卡表 | 不合法 |
|---|---|---|
| 2026-09-04 | 44 | 0 |
| 2026-09-05 | 80 | 0 |
| 2026-09-06 | 14 | 0 |
| 2026-09-07 | 63 | 0 |
| 2026-09-08 | 47 | 0 |
| 2026-09-09 | 68 | 0 |
| 2026-09-10 | 73 | 0 |
| 2026-09-11 | 64 | 0 |
| 2026-09-12 | 45 | 0 |
| 2026-09-13 | 34 | 0 |
| 2026-09-14 | 44 | 0 |
| 2026-09-15 | 29 | 0 |
| 2026-09-16 | 30 | 0 |
| 2026-09-17 | 36 | 0 |
| 2026-09-18 | 30 | 0 |
| 2026-09-19 | 30 | 0 |
| 2026-09-20 | 42 | 0 |
| 2026-09-21 | 36 | 0 |
| 2026-09-22 | 38 | 0 |
| 2026-09-23 | 1 | 0 |

## Yugipedia 关系

`relations.json` 覆盖卡池中 10932 张卡。各属性无法对应到卡片数据库的页面数：archetype_support 67、anti_support 6、archseries_related 23、archseries 104。
