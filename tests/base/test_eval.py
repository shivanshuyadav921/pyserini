#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import unittest

from pyserini.eval.trec_eval import trec_eval


class TestTrecEval(unittest.TestCase):
    def setUp(self):
        # The current directory depends on if you're running inside an IDE or from command line.
        curdir = os.getcwd()
        if curdir.endswith("core"):
            self.root = "../../"
        else:
            self.root = "."

        self.qrels_path = os.path.join(
            self.root, "tools/topics-and-qrels/qrels.covid-round1.txt"
        )
        self.run_path = os.path.join(
            self.root, "tests/resources/simple_trec_run_filter.txt"
        )

    def test_aggeregated_scores(self):
        args = [
            "-c",
            "-m",
            "ndcg_cut.10",
            self.qrels_path,
            self.run_path,
        ]
        self.assertEqual(trec_eval(args), 0.055)

    def test_single_query_score(self):
        args = [
            "-c",
            "-q",
            "-m",
            "ndcg_cut.10",
            self.qrels_path,
            self.run_path,
        ]
        self.assertEqual(trec_eval(args, query_id="1"), 0.2201)

    def test_per_query_unaggeregated_scores(self):
        args = [
            "-c",
            "-q",
            "-m",
            "ndcg_cut.10",
            self.qrels_path,
            self.run_path,
        ]
        expected = {"1": 0.2201, "2": 0.0, "3": 0.0, "4": 0.0, "all": 0.055}
        self.assertDictEqual(trec_eval(args, return_per_query_results=True), expected)

    def test_judged_at_k_scores(self):
        args = [
            "-c",
            "-q",
            "-m",
            "judged.20",
            self.qrels_path,
            self.run_path,
        ]
        self.assertEqual(trec_eval(args), 0.5)

    def test_judged_only_metric_does_not_crash(self):
        """Regression test for issue #2329: requesting *only* `-m judged.N` must not crash,
        and must return the judged.N value itself (not some unrelated leftover metric from
        trec_eval's default report).

        When every "-m" flag the caller passed is a Pyserini-only pseudo-metric (currently
        just "judged.N"), that flag is stripped before invoking the real trec_eval binary.
        If no other "-m" flag survives, trec_eval falls back to printing its full default
        report whose first line is a non-numeric "runid <TAB> all <TAB> <run_tag>" header.
        The old eager ``float()`` cast in the dict comprehension crashed on that line with:
            ValueError: could not convert string to float: 'Anserini'
        (see issue #2329 and the CLI invocation:
            python -m pyserini.eval.trec_eval -c -m judged.20 \\
                tools/topics-and-qrels/qrels.hc4-neuclir22-fa.test.txt \\
                runs/run.index.neuclir22-fa-en.test_title.bm25-default
        )
        """
        args = [
            "-c",
            "-m",
            "judged.20",
            self.qrels_path,
            self.run_path,
        ]
        # Should not raise. The exact expected value depends on the fixture at
        # tests/resources/simple_trec_run_filter.txt and the qrels.covid-round1.txt.
        result = trec_eval(args)
        self.assertIsInstance(result, float)

    def test_judged_with_other_metric_returns_requested_metric(self):
        """When judged.N is combined with a "real" trec_eval metric, the underlying binary
        still gets a valid `-m` flag, so this path mostly worked before -- this just locks
        in that the fix doesn't regress it.
        """
        args = [
            "-c",
            "-m",
            "ndcg_cut.10",
            "-m",
            "judged.20",
            self.qrels_path,
            self.run_path,
        ]
        result = trec_eval(args)
        self.assertIsInstance(result, float)

    def test_jar_directory_exists(self):
        import importlib.resources
        jar_directory = str(importlib.resources.files("pyserini") / "resources" / "jars")
        self.assertTrue(os.path.isdir(jar_directory))

if __name__ == "__main__":
    unittest.main()
