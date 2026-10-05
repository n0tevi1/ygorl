# 剩余起手可达性调查（2026-10-05）

原320手跨独立诊断有273手完整可达、14手限域规则解释、33手仍unknown。
本轮容量/教师标签修复没有改变这个分类。继续核对资源受限的原失败起手，先检查Resonators11，
再按同一方法检查Kewl-Tune7、Maliss0、Yummy2；这些案例依据其原始起手选择，不加入训练或回填旧gate。

协议：从原ordinary-coverage精确start、相同环境/目标，在host穷举至turn2；分别使用snapshot逐枝恢复和
逐节点独立冷重放，比较完整节点/叶子及响应路径。只排除手牌重排和**host明确标成undo的cancel**，
保留效果处理中合法可选cancel。无beam、状态去重、启发式动作裁剪或成功率抽样。
深度80、200,000节点、600秒均为硬guard：任何guard命中只能记录incomplete/unknown，不能称不可达。
发现目标则保存完整路径并独立重放、健康验证；树完成且目标从未出现，才能给这个明确有限域下的不可达证据。
所有节点检查error/retry/Lua/未知/不可解码消息；真实卡文/脚本/核心/source身份与原记录hash一起保存。
排除重排/无事件撤销的域限制必须随结论一起报告，不将有限任务等同完整对局强度。

修正标签后的128×2检查点在容量validation中达成了一个原30秒教师未解出的起手（Ryzeal-Mitsurugi7）。
先独立保存其完整host动作/响应并验真，只作为搜索漏解见证，不回填已冻结验证集。
另把这一个已完成的固定检查点以argmax运行在原33个unknown上，保留全部结果；这是已暴露开发案例的
可达性诊断，不是新的策略强度评估。只有完整健康重放到turn2的成功才可变更“已知可达”分类。
不据结果挑随机种子/epoch，也不把新增见证加入本轮训练。

## 结果

Resonators11 的两套枚举均完成 **20,306节点 /4,581 turn2叶子**；Kewl-Tune7 均完成
**2,111节点 /446叶子**。snapshot 与逐节点冷重放的全部路径、动作、场面逐项相等，
全程无健康错误、无达标场面。原始数据、卡文和脚本身份随树保存；不是只对照节点数量。

Resonators11 起手为两张 Dominus Impulse、Red Dragon Archfiend Chain、Imperm、Nibiru。
Chain 可以展示额外卡组的 Red Dragon Archfiend 后在盖放回合发动，但仍需要对手效果怪兽目标；
被动对手开局没有该目标。Nibiru 所需的对手五次召唤亦不存在；整棵树没有可发动/召唤动作。
Kewl-Tune7 为两张 Ash、Imperm、Veiler、Ghost Ogre，完整树只有通常召唤/盖放等分支，
无特殊召唤/发动资源可到要求的同调目标。两例结论仅限本协议的目标和被动开局。

Maliss0 **200,000节点 /32,939叶子**、Yummy2 **200,000节点 /36,732叶子**均触及节点guard，
分别329.7s/443.3s，未启动无必要的冷重放。这两例继续unknown；计数不是不可达证明。
修正128×2在原33unknown上 **0恢复**。原320的当前诊断分类因此为
**273完整可达 /16限域不可达解释 /31unknown**，未通过原覆盖gate，也未改动冻结数据。

另一套容量validation的 Ryzeal-Mitsurugi7 已独立完整重放通过：111个host动作、110响应、
agent0 84步，健康到turn2。它是原30秒教师漏解的具体见证，不能计入原320分类。
文件 `capacity-validation-ryzeal-mitsurugi-07-policy-witness.json` 的 SHA-256 为
`1f25bda07832f7fed56ab82138ac8611d1612659f6fc1aaf88b96a5011176857`；
来源标为policy-only，solver_steps/solver_responses=0，不伪装成原生前缀或加入本轮训练。

工件目录 `out/research/remaining-reachability-2026-10-05/`，封存见 `validation.json`。
首次见证脚本误以为 `verify_line` 返回场面，修正其实际返回None的API使用后完整重跑；
原失败日志保留，不属于引擎/模型失败。
