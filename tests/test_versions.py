"""Test offline del riconoscimento di versioni più recenti: python3 -m unittest discover -s tests -t ."""
import unittest

from radar.models import _is_newer, parse_version as pv


class TestVersions(unittest.TestCase):
    def test_parse(self):
        g = pv("ibm-granite/granite-4.1-8b")
        self.assertEqual((g["fam"], g["ver"], g["size"], g["suf"]), ("granite", (4, 1), "8b", ""))
        q = pv("Qwen/Qwen3.8-27B")
        self.assertEqual((q["fam"], q["ver"], q["size"]), ("qwen", (3, 8), "27b"))
        d = pv("deepseek-ai/DeepSeek-V4.1-Flash")
        self.assertEqual((d["fam"], d["ver"], d["suf"]), ("deepseek", (4, 1), "flash"))
        l = pv("LiquidAI/LFM2.5-24B-A2B")
        self.assertEqual((l["fam"], l["ver"], l["size"]), ("lfm", (2, 5), "24b"))
        self.assertIsNone(pv("openai/gpt-oss-120b"))
        self.assertIsNone(pv("Muse Ember"))

    def test_newer(self):
        self.assertTrue(_is_newer(pv("granite-4.1-8b"), pv("granite-4.2-8b")))
        self.assertFalse(_is_newer(pv("granite-4.1-8b"), pv("granite-4.2-30b")))       # taglia diversa
        self.assertFalse(_is_newer(pv("granite-4.1-8b"), pv("granite-switch-4.2-8b")))  # famiglia diversa
        self.assertTrue(_is_newer(pv("GPT-6 Luna"), pv("GPT-6.1 Luna")))
        self.assertFalse(_is_newer(pv("GPT-6 Luna"), pv("GPT-6.1 Terra")))              # linea diversa
        self.assertFalse(_is_newer(pv("DeepSeek-V4.1-Flash"), pv("DeepSeek-V4.1-Flash-1017")))  # stessa versione
        self.assertTrue(_is_newer(pv("Claude Fable 5.1"), pv("Claude Fable 5.2")))
        self.assertFalse(_is_newer(pv("LFM2.5-24B-A2B"), pv("LFM3-1.7B")))


if __name__ == "__main__":
    unittest.main()
