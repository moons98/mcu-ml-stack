"""Res-CNN-LSTM (random weights) -> int8 TFLite -> C array.

No training: this model is only used for latency/memory benchmarking, and compute cost
and tensor shapes do not depend on weight values.
- Calibration: a subset of x_train from a preprocessed SEU npz if --calib is given,
  otherwise uniform random values in [0, 1] (same range as the MinMaxScaler output).
- Input/output tensors are int8 as well (used directly on the MCU, no float conversion).

Outputs (out/): res_cnn_lstm_{fp32,int8}.tflite, model_data.{cc,h}, export.json
The firmware (firmware/tflm_bench) compiles out/model_data.cc directly at build time.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE))
from model import build_res_cnn_lstm  # noqa: E402


def to_c_array(data: bytes, name: str) -> str:
    lines = []
    for i in range(0, len(data), 12):
        lines.append("  " + ", ".join(f"0x{b:02x}" for b in data[i:i + 12]) + ",")
    body = "\n".join(lines)
    return (f'#include "{name}.h"\n\n'
            f"alignas(16) const unsigned char g_{name}[] = {{\n{body}\n}};\n"
            f"const unsigned int g_{name}_len = {len(data)};\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--calib", type=Path, default=None, help="preprocessed SEU npz (optional)")
    ap.add_argument("--n-calib", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    model = build_res_cnn_lstm()
    assert model.count_params() == 4598, model.count_params()
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / "res_cnn_lstm_random.h5")

    rng = np.random.default_rng(args.seed)
    if args.calib:
        x = np.load(args.calib)["x_train"]
        calib = x[rng.choice(len(x), args.n_calib, replace=False)]
    else:
        calib = rng.uniform(0, 1, size=(args.n_calib, 16, 6)).astype(np.float32)

    def rep_data():
        for i in range(len(calib)):
            yield [calib[i:i + 1]]

    # A fixed batch-1 input signature is required for the LSTM to be fused into UNIDIRECTIONAL_SEQUENCE_LSTM
    run = tf.function(lambda x: model(x))
    concrete = run.get_concrete_function(tf.TensorSpec([1, 16, 6], tf.float32))

    results = {}
    for tag, quant in (("fp32", False), ("int8", True)):
        conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
        if quant:
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = rep_data
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type = tf.int8
            conv.inference_output_type = tf.int8
        tfl = conv.convert()
        path = args.out / f"res_cnn_lstm_{tag}.tflite"
        path.write_bytes(tfl)

        interp = tf.lite.Interpreter(model_content=tfl, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        interp.allocate_tensors()
        ops = [d["op_name"] for d in interp._get_ops_details()]
        results[tag] = {"size_bytes": len(tfl), "ops": ops}
        print(f"[{tag}] {len(tfl)} bytes ({len(tfl)/1024:.1f} KB)")
        print(f"  ops ({len(ops)}): {ops}")
        if quant:
            i_d, o_d = interp.get_input_details()[0], interp.get_output_details()[0]
            results[tag]["input"] = {"shape": i_d["shape"].tolist(), "dtype": str(i_d["dtype"]),
                                     "scale": float(i_d["quantization"][0]), "zero_point": int(i_d["quantization"][1])}
            results[tag]["output"] = {"shape": o_d["shape"].tolist(), "dtype": str(o_d["dtype"]),
                                      "scale": float(o_d["quantization"][0]), "zero_point": int(o_d["quantization"][1])}
            print(f"  input {results[tag]['input']}\n  output {results[tag]['output']}")

    # Sanity check: compare int8 TFLite output against the Keras float model
    interp = tf.lite.Interpreter(model_content=(args.out / "res_cnn_lstm_int8.tflite").read_bytes())
    interp.allocate_tensors()
    i_d, o_d = interp.get_input_details()[0], interp.get_output_details()[0]
    agree, n = 0, 200
    for k in range(n):
        xk = calib[k % len(calib)][None]
        q = np.round(xk / i_d["quantization"][0] + i_d["quantization"][1]).clip(-128, 127).astype(np.int8)
        interp.set_tensor(i_d["index"], q)
        interp.invoke()
        yq = (interp.get_tensor(o_d["index"]).astype(np.float32) - o_d["quantization"][1]) * o_d["quantization"][0]
        agree += int(yq.argmax() == model(xk).numpy().argmax())
    results["int8_vs_float_argmax_agreement"] = agree / n
    print(f"int8 vs float argmax agreement: {agree}/{n}")

    # Emit the model as a C array
    data = (args.out / "res_cnn_lstm_int8.tflite").read_bytes()
    (args.out / "model_data.cc").write_text(to_c_array(data, "model_data"))
    (args.out / "model_data.h").write_text(
        "#pragma once\n\nextern const unsigned char g_model_data[];\nextern const unsigned int g_model_data_len;\n")
    (args.out / "export.json").write_text(json.dumps(results, indent=2))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
