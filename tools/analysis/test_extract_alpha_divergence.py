import ast
import codecs
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import keras
import tensorflow as tf
import pandas as pd


SCRIPT = Path(__file__).with_name("extract_alpha_divergence.py")
spec = importlib.util.spec_from_file_location("extract_alpha_divergence", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class AlphaAnalysisTests(unittest.TestCase):
    def test_default_command_requires_hierarchical_training_output(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(analysis, 'BASE_DIR', directory), \
                 patch('sys.argv', ['extract_alpha_divergence.py']):
                with self.assertRaisesRegex(FileNotFoundError, 'train.py --full_training'):
                    analysis.main()

    def test_default_command_writes_next_to_hierarchical_training_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            version = 'full_model_hierarchical_source_filter_v1'
            csv_path = Path(directory) / 'results' / 'alpha_analysis' / f'{version}_alpha_per_sample.csv'
            csv_path.parent.mkdir(parents=True)
            csv_path.touch()
            with patch.object(analysis, 'BASE_DIR', directory), \
                 patch.object(analysis, 'DEFAULT_OUT_ROOT', str(csv_path.parent)), \
                 patch('sys.argv', ['extract_alpha_divergence.py']), \
                 patch.object(analysis, 'analyze_main_model_csv') as analyze:
                analysis.main()
            self.assertEqual(Path(analyze.call_args.args[1]),
                             csv_path.parent / version / 'main_model_analysis')

    def test_output_file_encodings_are_valid(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        encodings = [
            ast.literal_eval(keyword.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "encoding"
        ]
        for encoding in encodings:
            with self.subTest(encoding=encoding):
                codecs.lookup(encoding)

    def test_summary_report_formats_subject_statistics(self):
        result = {
            "subj_table": pd.DataFrame({"label": [0, 1]}),
            "aug0_df": pd.DataFrame({"label": [0, 1]}),
            "table15": pd.DataFrame([{
                "metric": "alpha1", "hc_median": 1.0, "hc_q1": 0.9, "hc_q3": 1.1,
                "pd_median": 1.2, "pd_q1": 1.1, "pd_q3": 1.3,
                "mannwhitney_u": 1.0, "p_value": 0.3, "cliffs_delta": 0.5,
            }]),
            "corr_df": pd.DataFrame(),
            "icc1": 0.3, "icc2": 0.4, "p_chi": 0.5,
            "dom_hc": pd.Series([True]), "dom_pd": pd.Series([False]),
            "dom_acc": pd.DataFrame(),
        }
        report = analysis.build_summary_report(result, 0.8, 0.7, 0.6, 0.9, 2, 42, "pcgita")
        self.assertIn("alpha1", report)

    def test_console_can_print_report_with_special_characters(self):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="gbk")
        analysis.configure_utf8_console(stream)
        stream.write("sample-level: F0-to-MFCC \u2011")
        stream.flush()
        self.assertEqual(stream.encoding, "utf-8")

    def test_skip_train_rejects_checkpoint_without_corrected_path_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "model_protocol.json"
            with self.assertRaisesRegex(RuntimeError, "retrain"):
                analysis.require_corrected_model_protocol(manifest)
            manifest.write_text(json.dumps({"alpha_path_version": "legacy"}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "retrain"):
                analysis.require_corrected_model_protocol(manifest)
            manifest.write_text(json.dumps({"alpha_path_version": analysis.ALPHA_PATH_VERSION}), encoding="utf-8")
            analysis.require_corrected_model_protocol(manifest)

    def test_help_prints_on_windows_gbk_console(self):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "gbk"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            capture_output=True, text=True, encoding="gbk", errors="replace", env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--filename-regex", result.stdout)

    def test_default_filename_pattern_matches_existing_recording(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        arguments = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and "filename" in node.args[0].value
        ]
        self.assertEqual(len(arguments), 1)
        arg = arguments[0]
        self.assertEqual(arg.args[0].value, "--filename-regex")
        pattern = ast.literal_eval(next(k.value for k in arg.keywords if k.arg == "default"))
        self.assertIsNotNone(re.fullmatch(pattern, "AVPEPUDEA0001e1_aug0"))

    def test_classification_path_has_gradient_to_alpha_network(self):
        tf.random.set_seed(42)
        layer = analysis.ModalCrossAttention(embed_dim=16, num_heads=2, dropout_rate=0)
        mfcc = tf.random.normal((4, 16))
        f0 = tf.random.normal((4, 16))
        with tf.GradientTape() as tape:
            mfcc_out, f0_out = layer(mfcc, f0, training=False)
            loss = tf.reduce_sum(tf.square(mfcc_out - f0_out))
        gradients = tape.gradient(loss, layer.alpha_net.trainable_variables)
        nonzero = [
            float(tf.reduce_max(tf.abs(g)).numpy())
            for g in gradients if g is not None
        ]
        self.assertTrue(nonzero and max(nonzero) > 1e-8, nonzero)

    def test_path_override_changes_output_and_can_be_reset(self):
        tf.random.set_seed(43)
        layer = analysis.ModalCrossAttention(embed_dim=16, num_heads=2, dropout_rate=0)
        mfcc = tf.random.normal((4, 16))
        f0 = tf.random.normal((4, 16))
        original = layer(mfcc, f0, training=False)
        layer.set_alpha_override(alpha1=0.0)
        ablated = layer(mfcc, f0, training=False)
        self.assertGreater(
            float(tf.reduce_max(tf.abs(original[0] - ablated[0])).numpy()), 1e-6
        )
        layer.set_alpha_override()
        restored = layer(mfcc, f0, training=False)
        self.assertLess(
            float(tf.reduce_max(tf.abs(original[0] - restored[0])).numpy()), 1e-6
        )

    def test_path_ablation_returns_predictions_and_restores_model(self):
        tf.random.set_seed(44)
        mfcc_input = keras.Input(shape=(16,))
        f0_input = keras.Input(shape=(16,))
        path = analysis.ModalCrossAttention(embed_dim=16, num_heads=2, dropout_rate=0)
        mfcc_out, f0_out = path(mfcc_input, f0_input)
        output = keras.layers.Dense(1, activation="sigmoid")(
            keras.layers.Concatenate()([mfcc_out, f0_out]))
        model = keras.Model([mfcc_input, f0_input], output)
        inputs = [tf.random.normal((4, 16)), tf.random.normal((4, 16))]
        baseline = model.predict(inputs, verbose=0).ravel()
        without_path1, without_path2 = analysis.predict_path_ablations(
            model, path, inputs, batch_size=4)
        self.assertEqual(without_path1.shape, (4,))
        self.assertEqual(without_path2.shape, (4,))
        self.assertTrue((abs(baseline - without_path1) > 1e-7).any())
        self.assertTrue((abs(baseline - without_path2) > 1e-7).any())
        self.assertEqual(path.alpha_override.numpy().tolist(), [-1.0, -1.0])
        self.assertTrue((abs(baseline - model.predict(inputs, verbose=0).ravel()) < 1e-6).all())

    def test_main_model_csv_analysis_uses_subject_level_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'alpha.csv'
            out = Path(directory) / 'analysis'
            rows = []
            for label, prefix in [(0, 'HC'), (1, 'PD')]:
                for subject in range(4):
                    for task in ('e1', 'e2'):
                        rows.append({
                            'split': 'test', 'filename': f'{prefix}{subject:02d}{task}_aug0.wav',
                            'subject': f'{prefix}{subject:02d}', 'label': label,
                            'alpha_mfcc_to_f0': 0.3 + 0.1 * label,
                            'alpha_f0_to_mfcc': 0.7 - 0.1 * label,
                            'prediction': 0.4 + 0.1 * label,
                            'delta_mfcc_to_f0': 0.01,
                            'delta_f0_to_mfcc': 0.02,
                            'source_f0_median_hz': 180 + 2 * subject,
                            'source_f0_change_hz': 2 + subject,
                            'source_voicing_probability': 0.7 + 0.01 * subject,
                        })
            pd.DataFrame(rows).to_csv(source, index=False)
            self.assertTrue(hasattr(analysis, 'analyze_main_model_csv'))
            analysis.analyze_main_model_csv(source, out)
            self.assertTrue((out / 'subject_balance_statistics.csv').exists())
            self.assertTrue((out / 'alpha_divergence.png').exists())
            self.assertTrue((out / 'source_alpha_relationship.png').exists())
            self.assertTrue((out / 'path_effects.png').exists())
            self.assertTrue((out / 'source_alpha_correlations.csv').exists())
            self.assertTrue((out / 'path_diagnostics.json').exists())
            self.assertEqual(len(pd.read_csv(out / 'subject_balance_statistics.csv')), 8)


if __name__ == "__main__":
    unittest.main()
