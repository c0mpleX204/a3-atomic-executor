# Third-party notices

`a3_atomic_executor/arm_executor.py` adapts A3 controller code from [Eter0109/a3_dual_arm_sim](https://github.com/Eter0109/a3_dual_arm_sim), copyright © 2026 Eter0109, under the MIT license. The original copyright and license terms are included in [LICENSE](LICENSE).

The A3 simulator, robot descriptions, and meshes are supplied by a separate checkout. Follow the upstream license and asset provenance notices for that checkout. Those assets are outside the scope of this repository's MIT license.

The CLAP experiment calls [Generalizable CLAP](https://github.com/Jianshu-Hu/Generalizable-CLAP) at its pinned upstream revision. The bundled `clap_a3/weights/config.yaml` and checkpoint parts come from the authors' official [pretrained checkpoint release](https://github.com/Jianshu-Hu/Generalizable-CLAP/blob/e536857194288d4707bbdce9075ce855e3eb4b49/scripts/EVALUATION.md#pre-trained-ckpts). See [the upstream Apache 2.0 license](clap_a3/UPSTREAM-LICENSE) and [the checkpoint manifest](clap_a3/weights/manifest.json).
