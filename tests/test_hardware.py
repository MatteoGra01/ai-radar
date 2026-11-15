"""Test offline dello stimatore VRAM: python3 -m unittest discover -s tests"""
import unittest

from radar.hardware import active_from_arch, arch_from_config, estimate, kv_bytes, params_from_name

HW = {"gpu_name": "RTX 5090", "vram_gb": 32, "reserved_gb": 1.0, "margin_gb": 1.5, "bandwidth_gbs": 1792,
      "system_ram_gb": 64, "ref_context": 32768, "contexts": [8192, 32768, 131072]}

# struttura reale di Qwen3.8-27B (attenzione ibrida: 1 layer full ogni 4)
HYBRID = {"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {
    "num_hidden_layers": 64, "hidden_size": 5120, "num_attention_heads": 24, "num_key_value_heads": 4,
    "head_dim": 256, "full_attention_interval": 4, "linear_num_value_heads": 48, "linear_key_head_dim": 128,
    "linear_value_head_dim": 128, "max_position_embeddings": 262144, "vocab_size": 248320, "intermediate_size": 17408}}

MOE = {"num_hidden_layers": 48, "hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 4,
       "head_dim": 128, "num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 768,
       "max_position_embeddings": 131072, "vocab_size": 151936, "intermediate_size": 6144}


class TestHardware(unittest.TestCase):
    def test_names(self):
        self.assertEqual(params_from_name("Qwen/Qwen3.6-35B-A3B"), (35.0, 3.0))
        self.assertEqual(params_from_name("Qwen/Qwen3.8-2.4T-A95B"), (2400.0, 95.0))
        self.assertEqual(params_from_name("Qwen/Qwen3-ASR-0.6B-hf"), (0.6, None))
        self.assertEqual(params_from_name("moonshotai/Kimi-K3.1-Pro"), (None, None))

    def test_hybrid_kv(self):
        a = arch_from_config(HYBRID)
        self.assertEqual((a["full_layers"], a["linear_layers"]), (16, 48))
        # 16 layer × 2 × 4 kv × 256 dim × 2 byte = 64 KiB/token (+ stato costante dei layer lineari)
        per_tok = kv_bytes(a, 2) - kv_bytes(a, 1)
        self.assertEqual(per_tok, 16 * 2 * 4 * 256 * 2)

    def test_verdict_27b(self):
        a = arch_from_config(HYBRID)
        v = estimate(27.78, None, a, HW, "image-text-to-text")
        self.assertEqual(v["level"], "ok")
        # Q8_0 entra solo al limite a 32k: la consigliata è Q6_K, e il verdetto deve dire che serve quantizzare
        self.assertEqual(v["best_quant"], "Q6_K")
        self.assertEqual(v["tight"], "Q8_0")
        self.assertTrue(v["quantized"])
        self.assertIn("Serve quantizzazione", v["text"])
        q4 = next(r for r in v["table"] if r["q"] == "Q4_K_M")
        self.assertAlmostEqual(q4["weights_gb"], 15.7, delta=0.2)
        self.assertGreater(q4["max_ctx"], 200_000)

    def test_moe(self):
        a = arch_from_config(MOE)
        act = active_from_arch(a, 30.5)
        self.assertTrue(2.5 < act < 4.5, act)
        v = estimate(30.5, act, a, HW, "text-generation")
        self.assertTrue(v["is_moe"])
        self.assertEqual(v["level"], "ok")

    def test_too_big_and_offload(self):
        self.assertEqual(estimate(700, 40, None, HW)["level"], "no")
        v = estimate(120, 5, None, HW)  # MoE grande con pochi attivi: offload esperti in RAM
        self.assertEqual(v["level"], "warn")
        self.assertIn("offload", v["text"])

    def test_bad_config(self):
        self.assertIsNone(arch_from_config({"sliding_window_pattern": "interleave"}))
        self.assertIsNotNone(arch_from_config({**MOE, "sliding_window_pattern": "interleave", "sliding_window": 4096}))


if __name__ == "__main__":
    unittest.main()
